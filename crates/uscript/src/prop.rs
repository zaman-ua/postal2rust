//! Property handles for host code that reads the same properties every frame.
//!
//! [`Vm::get_prop`] finds a property by its name: even with a cache that means hashing and a string
//! compare, thousands of times per tick. A [`Prop`] is one call site's name with a small number of
//! its own; every class remembers which slot that number resolves to (or that the class has no such
//! property), so after the first use a lookup is two array reads.
//!
//! ```ignore
//! let hidden = vm.field(actor, prop!("bHidden")).is_some_and(|v| v.as_bool());
//! vm.set_field(actor, prop!("Location"), Value::Vector(location));
//! ```
//!
//! Use `get_prop` / `set_prop` for names only known at run time, and wherever speed is irrelevant.

use crate::vm::{ClassInfo, ObjId, Value, Vm};
use std::cell::RefCell;
use std::sync::atomic::{AtomicUsize, Ordering};

/// A property name with a process-wide number, made by [`prop!`](crate::prop!).
pub struct Prop {
    name: &'static str,
    /// 0 until first use, then the number plus one.
    key: AtomicUsize,
}

impl Prop {
    pub const fn new(name: &'static str) -> Prop {
        Prop { name, key: AtomicUsize::new(0) }
    }

    pub fn name(&self) -> &'static str {
        self.name
    }

    fn key(&self) -> usize {
        match self.key.load(Ordering::Relaxed) {
            0 => {
                static NEXT: AtomicUsize = AtomicUsize::new(1);
                let new = NEXT.fetch_add(1, Ordering::Relaxed);
                // Another thread may have numbered it meanwhile: its number wins.
                match self.key.compare_exchange(0, new, Ordering::Relaxed, Ordering::Relaxed) {
                    Ok(_) => new - 1,
                    Err(theirs) => theirs - 1,
                }
            }
            n => n - 1,
        }
    }
}

/// A `&'static Prop` for a literal property name, one per call site.
#[macro_export]
macro_rules! prop {
    ($name:literal) => {{
        static PROP: $crate::prop::Prop = $crate::prop::Prop::new($name);
        &PROP
    }};
}

const UNKNOWN: u32 = 0;
const ABSENT: u32 = 1;
/// Entries from here on are `slot + FIRST_SLOT`.
const FIRST_SLOT: u32 = 2;

/// A class's answers by [`Prop`] number. Lives in the [`ClassInfo`], whose slots never change, so
/// it is dropped with the class and cannot go stale.
#[derive(Debug, Clone, Default)]
pub struct PropCache(RefCell<Vec<u32>>);

impl ClassInfo {
    /// [`ClassInfo::slot`] through a property handle.
    pub fn field_slot(&self, prop: &Prop) -> Option<usize> {
        let key = prop.key();
        let known = self.prop_cache.0.borrow().get(key).copied().unwrap_or(UNKNOWN);
        match known {
            UNKNOWN => {
                let slot = self.slot(prop.name);
                let mut cache = self.prop_cache.0.borrow_mut();
                if cache.len() <= key {
                    cache.resize(key + 1, UNKNOWN);
                }
                cache[key] = slot.map_or(ABSENT, |s| s as u32 + FIRST_SLOT);
                slot
            }
            ABSENT => None,
            entry => Some((entry - FIRST_SLOT) as usize),
        }
    }
}

impl Vm {
    /// [`Vm::get_prop`] through a property handle.
    pub fn field(&self, obj: ObjId, prop: &Prop) -> Option<&Value> {
        let object = self.objects.get(obj)?;
        object.slots.get(self.classes[object.class].field_slot(prop)?)
    }

    /// [`Vm::set_prop`] through a property handle.
    pub fn set_field(&mut self, obj: ObjId, prop: &Prop, value: Value) -> bool {
        let class = self.objects[obj].class;
        match self.classes[class].field_slot(prop) {
            Some(slot) => {
                self.objects[obj].slots[slot] = value;
                true
            }
            None => false,
        }
    }
}
