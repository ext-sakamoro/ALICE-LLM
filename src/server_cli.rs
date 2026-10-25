//! Command-line options of the `alice-llm-server` binary, and the stdin
//! watch that ties the server's life to the process that started it.
//!
//! Kept in the library (std only, no feature gate) so `cargo test --lib`
//! covers it: the binary needs the `server` feature and a model to start.
//!
//! The server listens on the loopback address unless `--host` names another
//! one. An application that bundles the server talks to it over loopback, so
//! the default exposes nothing to the network; listening on every interface
//! is an explicit choice (`--host 0.0.0.0`).
//!
//! `--exit-on-stdin-eof` makes the server exit when its stdin closes. A parent
//! that spawns the server with a piped stdin and keeps the write end open gets
//! the server stopped however the parent ends (normal exit, a signal, or a
//! crash), because the operating system closes the pipe with the parent. The
//! watch starts before the model is read, so a parent that dies during a long
//! load does not leave the server behind either.

use std::io::Read;
use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::thread::JoinHandle;

/// Address the server binds when `--host` is not given
pub const DEFAULT_HOST: IpAddr = IpAddr::V4(Ipv4Addr::LOCALHOST);

/// Port the server binds when `--port` is not given
pub const DEFAULT_PORT: u16 = 8000;

/// Usage line printed for a missing or invalid argument
pub const USAGE: &str = "Usage: alice-llm-server --model <path.gguf> [--host 127.0.0.1] [--port 8000] [--hybrid] [--exit-on-stdin-eof]";

/// Parsed command line of `alice-llm-server`
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ServerOptions {
    /// GGUF model path
    pub model: String,
    /// Address to listen on (loopback unless `--host` is given)
    pub host: IpAddr,
    /// Port to listen on
    pub port: u16,
    /// Run inference on the CPU instead of the GPU
    pub hybrid: bool,
    /// Exit when stdin reaches end of file
    pub exit_on_stdin_eof: bool,
}

impl ServerOptions {
    /// Socket address the server binds
    #[must_use]
    pub const fn bind_addr(&self) -> SocketAddr {
        SocketAddr::new(self.host, self.port)
    }
}

/// Bind the listening socket for `opts`
///
/// The server binds before it reads the model, so a port in use or a refused
/// address is reported at once rather than after the load, and the address
/// it prints is the one the socket actually has.
///
/// # Errors
///
/// Returns the error of the underlying `bind`.
pub fn bind(opts: &ServerOptions) -> std::io::Result<std::net::TcpListener> {
    std::net::TcpListener::bind(opts.bind_addr())
}

/// The line the server prints to stdout right after it binds.
///
/// It names the address actually bound, and is a contract: a parent process that
/// spawns the server with `--port 0` learns the chosen port by reading it back.
#[must_use]
pub fn bound_report(addr: SocketAddr) -> String {
    format!("Bound to http://{addr}")
}

/// Value that follows `flag`, or an error when the flag is last
fn value_of<'a>(args: &'a [String], flag: &str) -> Result<Option<&'a str>, String> {
    match args.iter().position(|a| a == flag) {
        None => Ok(None),
        Some(i) => args
            .get(i + 1)
            .map(|v| Some(v.as_str()))
            .ok_or_else(|| format!("{flag} needs a value")),
    }
}

/// Parse the arguments after the program name
///
/// # Errors
///
/// Returns a message when `--model` is missing, a flag has no value, or
/// `--host` / `--port` does not parse.
pub fn parse_args(args: &[String]) -> Result<ServerOptions, String> {
    let model = value_of(args, "--model")?
        .ok_or_else(|| "--model is required".to_string())?
        .to_string();
    let host = match value_of(args, "--host")? {
        None => DEFAULT_HOST,
        Some(v) => v
            .parse::<IpAddr>()
            .map_err(|e| format!("--host {v}: {e}"))?,
    };
    let port = match value_of(args, "--port")? {
        None => DEFAULT_PORT,
        Some(v) => v.parse::<u16>().map_err(|e| format!("--port {v}: {e}"))?,
    };
    Ok(ServerOptions {
        model,
        host,
        port,
        hybrid: args.iter().any(|a| a == "--hybrid"),
        exit_on_stdin_eof: args.iter().any(|a| a == "--exit-on-stdin-eof"),
    })
}

/// Read `input` until end of file or a read error, then call `on_eof`
///
/// The server passes stdin and `std::process::exit`. Bytes read are
/// discarded: the pipe carries no messages, only its open or closed state.
pub fn watch_until_eof<R, F>(mut input: R, on_eof: F) -> std::io::Result<JoinHandle<()>>
where
    R: Read + Send + 'static,
    F: FnOnce() + Send + 'static,
{
    std::thread::Builder::new()
        .name("stdin-eof-watch".to_string())
        .spawn(move || {
            let mut buf = [0u8; 256];
            loop {
                match input.read(&mut buf) {
                    Ok(0) => break,
                    Ok(_) => {}
                    Err(e) if e.kind() == std::io::ErrorKind::Interrupted => {}
                    Err(_) => break,
                }
            }
            on_eof();
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use std::sync::mpsc;
    use std::time::Duration;

    fn args(list: &[&str]) -> Vec<String> {
        list.iter().map(|s| (*s).to_string()).collect()
    }

    #[test]
    fn default_bind_is_loopback() {
        let o = parse_args(&args(&["--model", "m.gguf"])).unwrap();
        assert_eq!(o.bind_addr(), "127.0.0.1:8000".parse().unwrap());
        assert!(o.host.is_loopback());
        assert!(!o.exit_on_stdin_eof);
        assert!(!o.hybrid);
    }

    #[test]
    fn host_and_port_are_taken_only_when_given() {
        let o = parse_args(&args(&[
            "--model", "m.gguf", "--host", "0.0.0.0", "--port", "9001",
        ]))
        .unwrap();
        assert_eq!(o.bind_addr(), "0.0.0.0:9001".parse().unwrap());
        let v6 = parse_args(&args(&["--model", "m.gguf", "--host", "::1"])).unwrap();
        assert_eq!(v6.bind_addr(), "[::1]:8000".parse().unwrap());
    }

    #[test]
    fn bad_values_are_refused_not_defaulted() {
        assert!(parse_args(&args(&["--model", "m", "--host", "localhost"])).is_err());
        assert!(parse_args(&args(&["--model", "m", "--port", "70000"])).is_err());
        assert!(parse_args(&args(&["--model", "m", "--port"])).is_err());
        assert!(parse_args(&args(&["--port", "8000"])).is_err());
    }

    #[test]
    fn flags_are_read() {
        let o = parse_args(&args(&[
            "--hybrid",
            "--model",
            "m.gguf",
            "--exit-on-stdin-eof",
        ]))
        .unwrap();
        assert!(o.hybrid);
        assert!(o.exit_on_stdin_eof);
        assert_eq!(o.model, "m.gguf");
    }

    #[test]
    fn the_default_socket_is_bound_on_loopback() {
        let o = parse_args(&args(&["--model", "m.gguf", "--port", "0"])).unwrap();
        let listener = bind(&o).unwrap();
        let local = listener.local_addr().unwrap();
        assert!(local.ip().is_loopback(), "bound on {local}");
        assert_ne!(local.port(), 0);
        // the stdout contract: the reported line names the resolved, non-zero port
        // so a parent spawning with --port 0 can read it back
        let line = bound_report(local);
        assert_eq!(line, format!("Bound to http://{local}"));
        assert!(line.starts_with("Bound to http://"));
        assert!(!line.ends_with(":0"));
        let wide =
            parse_args(&args(&["--model", "m", "--host", "0.0.0.0", "--port", "0"])).unwrap();
        assert!(bind(&wide)
            .unwrap()
            .local_addr()
            .unwrap()
            .ip()
            .is_unspecified());
    }

    #[test]
    fn watch_fires_only_after_the_writer_closes() {
        let (reader, mut writer) = std::io::pipe().unwrap();
        let (tx, rx) = mpsc::channel();
        let handle = watch_until_eof(reader, move || tx.send(()).unwrap()).unwrap();
        writer.write_all(b"still here").unwrap();
        assert!(
            rx.recv_timeout(Duration::from_millis(300)).is_err(),
            "the watch fired while the writer was open"
        );
        drop(writer);
        rx.recv_timeout(Duration::from_secs(5))
            .expect("the watch did not fire after the writer closed");
        handle.join().unwrap();
    }
}
