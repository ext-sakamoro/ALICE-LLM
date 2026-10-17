//! Hash containers that expose no iteration order (ALICE-Law G13).
//!
//! `HashMap` / `HashSet` iterate in an order that differs between runs, so
//! clippy.toml bans them everywhere in the crate. The keyed-lookup sites that
//! need hashing (token tables, caches, visited sets) use [`LookupMap`] and
//! [`LookupSet`] instead: lookup, insertion and removal by key, a length, and
//! nothing that visits the entries. `Debug` prints the length only. This
//! module is the one place allowed to name the std hash containers, so no
//! code outside it can observe their order. tests/lookup_surface.rs fails
//! when an iteration method or an iterating trait is added here.
#![allow(clippy::disallowed_types)]

use std::borrow::Borrow;
use std::collections::hash_map::Entry;
use std::collections::{HashMap, HashSet};
use std::fmt;
use std::hash::Hash;

/// A hash map with keyed access only (no iteration).
#[derive(Clone)]
pub(crate) struct LookupMap<K, V>(HashMap<K, V>);

impl<K: Eq + Hash, V> LookupMap<K, V> {
    pub(crate) fn new() -> Self {
        Self(HashMap::new())
    }

    pub(crate) fn with_capacity(capacity: usize) -> Self {
        Self(HashMap::with_capacity(capacity))
    }

    pub(crate) fn get<Q>(&self, key: &Q) -> Option<&V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        self.0.get(key)
    }

    pub(crate) fn insert(&mut self, key: K, value: V) -> Option<V> {
        self.0.insert(key, value)
    }

    pub(crate) fn remove<Q>(&mut self, key: &Q) -> Option<V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        self.0.remove(key)
    }

    pub(crate) fn entry(&mut self, key: K) -> Entry<'_, K, V> {
        self.0.entry(key)
    }

    pub(crate) fn len(&self) -> usize {
        self.0.len()
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

impl<K: Eq + Hash, V: PartialEq> PartialEq for LookupMap<K, V> {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl<K: Eq + Hash, V: Eq> Eq for LookupMap<K, V> {}

impl<K: Eq + Hash, V> Default for LookupMap<K, V> {
    fn default() -> Self {
        Self::new()
    }
}

impl<K: Eq + Hash, V> FromIterator<(K, V)> for LookupMap<K, V> {
    /// Later pairs replace earlier ones with the same key, in the order of
    /// the input iterator.
    fn from_iter<I: IntoIterator<Item = (K, V)>>(iter: I) -> Self {
        Self(FromIterator::from_iter(iter))
    }
}

impl<K, V> fmt::Debug for LookupMap<K, V> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LookupMap")
            .field("len", &self.0.len())
            .finish_non_exhaustive()
    }
}

/// A hash set with membership access only (no iteration).
#[derive(Clone)]
pub(crate) struct LookupSet<T>(HashSet<T>);

impl<T: Eq + Hash> LookupSet<T> {
    pub(crate) fn new() -> Self {
        Self(HashSet::new())
    }

    pub(crate) fn contains<Q>(&self, value: &Q) -> bool
    where
        T: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        self.0.contains(value)
    }

    /// `true` when the value was not present.
    pub(crate) fn insert(&mut self, value: T) -> bool {
        self.0.insert(value)
    }

    pub(crate) fn remove<Q>(&mut self, value: &Q) -> bool
    where
        T: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        self.0.remove(value)
    }

    pub(crate) fn len(&self) -> usize {
        self.0.len()
    }
}

impl<T: Eq + Hash> PartialEq for LookupSet<T> {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl<T: Eq + Hash> Eq for LookupSet<T> {}

impl<T: Eq + Hash> Default for LookupSet<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Eq + Hash> FromIterator<T> for LookupSet<T> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        Self(FromIterator::from_iter(iter))
    }
}

impl<T> fmt::Debug for LookupSet<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LookupSet")
            .field("len", &self.0.len())
            .finish_non_exhaustive()
    }
}
