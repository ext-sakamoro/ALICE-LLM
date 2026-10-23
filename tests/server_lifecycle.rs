//! The real `alice-llm-server` binary stops when its stdin closes.
//!
//! The model path is a FIFO with no writer, so the server blocks while
//! reading the model: the process is alive and past argument parsing, and no
//! model file is needed. Closing stdin must then end it.
//!
//! The second test shows the exit comes from `--exit-on-stdin-eof` and not
//! from the FIFO or the closed pipe alone. The third reads the address the
//! server reports binding (before the model read) and connects to it.
#![cfg(all(unix, feature = "server"))]

use std::io::{BufRead, BufReader};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

struct Scratch(std::path::PathBuf);

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn fifo(tag: &str) -> (Scratch, std::path::PathBuf) {
    let dir = std::env::temp_dir().join(format!(
        "alice-llm-server-lifecycle-{tag}-{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("model.gguf");
    let status = Command::new("mkfifo").arg(&path).status().unwrap();
    assert!(status.success(), "mkfifo failed");
    (Scratch(dir), path)
}

/// Start the server and wait until it reports that it is reading the model;
/// returns the address it reported binding
fn start(model: &std::path::Path, extra: &[&str]) -> (Child, std::net::SocketAddr) {
    let mut child = Command::new(env!("CARGO_BIN_EXE_alice-llm-server"))
        .arg("--model")
        .arg(model)
        .args(extra)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .unwrap();
    // Lines arrive through a thread, so a server that never prints a line
    // (it blocks on the FIFO with stdout open) fails the test after a bound
    // wait instead of hanging it; the thread keeps draining the pipe
    let (tx, rx) = std::sync::mpsc::channel::<String>();
    let stdout = child.stdout.take().unwrap();
    std::thread::spawn(move || {
        for line in BufReader::new(stdout).lines().map_while(Result::ok) {
            let _ = tx.send(line);
        }
    });
    let deadline = Instant::now() + Duration::from_secs(10);
    let mut wait_for = |marker: &str| -> String {
        loop {
            let left = deadline.saturating_duration_since(Instant::now());
            match rx.recv_timeout(left) {
                Ok(line) => {
                    if let Some(rest) = line.strip_prefix(marker) {
                        return rest.trim().to_string();
                    }
                }
                Err(_) => {
                    let _ = child.kill();
                    let _ = child.wait();
                    panic!("the server did not print `{marker}` within 10 s");
                }
            }
        }
    };
    let addr: std::net::SocketAddr = wait_for("Bound to http://").parse().unwrap();
    wait_for("Loading:");
    std::thread::sleep(Duration::from_millis(300));
    assert!(
        child.try_wait().unwrap().is_none(),
        "the server exited while the model read should block it"
    );
    (child, addr)
}

fn exited_within(child: &mut Child, limit: Duration) -> bool {
    let deadline = Instant::now() + limit;
    while Instant::now() < deadline {
        if child.try_wait().unwrap().is_some() {
            return true;
        }
        std::thread::sleep(Duration::from_millis(50));
    }
    false
}

#[test]
fn closing_stdin_stops_the_server_with_the_flag() {
    let (_scratch, model) = fifo("flag");
    let (mut child, _) = start(&model, &["--exit-on-stdin-eof", "--port", "0"]);
    drop(child.stdin.take());
    let stopped = exited_within(&mut child, Duration::from_secs(10));
    if !stopped {
        let _ = child.kill();
        let _ = child.wait();
    }
    assert!(stopped, "the server kept running after its stdin closed");
}

#[test]
fn without_the_flag_closing_stdin_does_not_stop_it() {
    let (_scratch, model) = fifo("noflag");
    let (mut child, _) = start(&model, &["--port", "0"]);
    drop(child.stdin.take());
    let stopped = exited_within(&mut child, Duration::from_secs(2));
    let _ = child.kill();
    let _ = child.wait();
    assert!(!stopped, "the server stopped without --exit-on-stdin-eof");
}

#[test]
fn the_server_listens_on_loopback_by_default() {
    let (_scratch, model) = fifo("bind");
    let (mut child, addr) = start(&model, &["--port", "0", "--exit-on-stdin-eof"]);
    let reachable = std::net::TcpStream::connect_timeout(&addr, Duration::from_secs(2)).is_ok();
    drop(child.stdin.take());
    let stopped = exited_within(&mut child, Duration::from_secs(10));
    if !stopped {
        let _ = child.kill();
        let _ = child.wait();
    }
    assert!(addr.ip().is_loopback(), "the server bound {addr}");
    assert!(reachable, "nothing accepted on the reported address {addr}");
    assert!(stopped, "the server kept running after its stdin closed");
}
