//! Python access to mapper networks, exploration policies, and store observers.
//!
//! MultiOptPy: no counterpart (component API).
//!
//! This file is part of mopt-py (the Python interface of mopt-rs).
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use super::err;
use mopt_rs::core_infra::{json::JsonValue, rng::Rng};
use mopt_rs::workflow::mapper::{
    network_json, Candidate, ExplorationEnd, MapperObserver, MethodSettings, Network,
    PrioritySettings, StoreObserver, StoreRegistry,
};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::collections::BTreeMap;
use std::path::PathBuf;

/// A snapshot in reaction_network.json format, with its original run settings.
#[pyclass(name = "MapperNetwork", frozen)]
pub struct PyMapperNetwork {
    network: Network,
    run: JsonValue,
}

#[pymethods]
impl PyMapperNetwork {
    #[new]
    fn new(json: &str) -> PyResult<Self> {
        let (network, run) =
            network_json::from_json(&JsonValue::parse(json).map_err(err)?).map_err(err)?;
        Ok(Self { network, run })
    }

    /// Tasks proposed for a node, as dictionaries with a key, description and parameters.
    #[pyo3(signature=(node, method="afir_ts", options=None, top_n=1))]
    fn tasks<'py>(
        &self,
        py: Python<'py>,
        node: usize,
        method: &str,
        options: Option<BTreeMap<String, String>>,
        top_n: usize,
    ) -> PyResult<Vec<Bound<'py, PyDict>>> {
        let settings = MethodSettings {
            extra: options.unwrap_or_default(),
            top_n,
            ..Default::default()
        };
        let rule = super::mapper_plugins::parts()
            .methods
            .create(method, &settings)
            .map_err(err)?;
        let tasks = rule
            .tasks(self.network.node(node).map_err(err)?, &self.network)
            .map_err(err)?;
        let loads = py.import("json")?.getattr("loads")?;
        tasks
            .into_iter()
            .map(|t| {
                let d = PyDict::new(py);
                d.set_item("key", t.key)?;
                d.set_item("description", t.description)?;
                d.set_item(
                    "parameters",
                    loads.call1((t.parameters.to_string_compact(),))?,
                )?;
                Ok(d)
            })
            .collect()
    }

    /// Select a node from (node ID, visits, remaining tasks) using a mapper priority.
    #[pyo3(signature=(priority,candidates,*,temperature=298.15,job=0,seed=0,options=None))]
    fn select(
        &self,
        priority: &str,
        candidates: Vec<(usize, usize, usize)>,
        temperature: f64,
        job: usize,
        seed: u64,
        options: Option<BTreeMap<String, String>>,
    ) -> PyResult<Option<usize>> {
        let mut rule = super::mapper_plugins::parts()
            .priorities
            .create_with(
                priority,
                &PrioritySettings { temperature },
                &options.unwrap_or_default(),
            )
            .map_err(err)?;
        let mut pool = Vec::new();
        for (id, visits, untried) in candidates {
            let n = self.network.node(id).map_err(err)?;
            if untried > 0 {
                pool.push(Candidate {
                    node: id,
                    visits,
                    untried,
                    energy: n.free_energy.unwrap_or(n.energy),
                });
            }
        }
        if pool.is_empty() {
            return Ok(None);
        }
        Ok(rule
            .select(&self.network, &pool, job, &mut Rng::from_seed(seed))
            .map_err(err)?
            .map(|i| pool[i].node))
    }

    /// Notify the Rust store observer of a finished or interrupted exploration.
    #[pyo3(signature=(directory,*,stores=None,reason=None,interrupted=false))]
    fn save(
        &self,
        directory: PathBuf,
        stores: Option<Vec<String>>,
        reason: Option<String>,
        interrupted: bool,
    ) -> PyResult<()> {
        let registry = StoreRegistry::builtin();
        let stores = stores
            .unwrap_or_else(|| vec!["files".into()])
            .iter()
            .map(|s| registry.create(s))
            .collect::<Result<Vec<_>, _>>()
            .map_err(err)?;
        std::fs::create_dir_all(&directory)?;
        let mut observer = StoreObserver::new(&directory, stores, |_, end| {
            let mut run = self.run.clone();
            if let JsonValue::Object(fields) = &mut run {
                fields.insert(
                    "ended".into(),
                    end.reason
                        .clone()
                        .map_or(JsonValue::Null, JsonValue::String),
                );
                fields.insert("interrupted".into(), JsonValue::Bool(end.interrupted));
            }
            run
        });
        observer
            .on_finish(
                &self.network,
                &ExplorationEnd {
                    reason,
                    interrupted,
                },
            )
            .map_err(err)
    }
}
