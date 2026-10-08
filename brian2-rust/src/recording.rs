//! In-memory monitors for the bounded PoC. The simulation records raw values;
//! text formatting and file writes happen only after the final tick.
use super::*;

pub(super) struct Recording {
    samples: Vec<f64>,
    spikes: Vec<(u32, u32)>, // tick, neuron; both bounded by Model::validate
    max_spikes: usize,
}

impl Recording {
    pub(super) fn new(model: &Model) -> Result<Self> {
        let monitor = &model.definition.monitor;
        // State samples are bounded by validation; spikes retain a separate
        // runtime cap as the supported neuron-tick budget grows.
        let sample_count = model.run.steps * monitor.record.len() * monitor.variables.len();
        let max_spikes = (model.run.steps * model.instance.neuron_count).min(10_000_000);
        let mut samples = Vec::new();
        samples.try_reserve_exact(sample_count)?;
        Ok(Self {
            samples,
            spikes: Vec::new(),
            max_spikes,
        })
    }

    pub(super) fn sample(&mut self, states: &[Vec<f64>], variables: &[usize], indices: &[usize]) {
        // Preserve variable order and duplicate record indices exactly.
        for &i in indices {
            for &state in variables {
                self.samples.push(states[state][i]);
            }
        }
    }

    pub(super) fn spike(&mut self, tick: usize, neuron: usize) -> Result<()> {
        if self.spikes.len() == self.spikes.capacity() {
            check(
                self.spikes.len() < self.max_spikes,
                "spike recording budget exceeded",
            )?;
            // Grow on demand, capped by the validated one-spike/neuron/tick
            // bound. Silent networks do not allocate a worst-case spike array.
            let capacity = (self.spikes.len().max(512) * 2).min(self.max_spikes);
            self.spikes
                .try_reserve_exact(capacity - self.spikes.len())?;
        }
        self.spikes.push((tick as u32, neuron as u32));
        Ok(())
    }

    pub(super) fn metadata(&self) -> serde_json::Value {
        serde_json::json!({
            "mode": "memory",
            "state_values": self.samples.len(),
            "spike_events": self.spikes.len(),
            "state_capacity_bytes": self.samples.capacity() * size_of::<f64>(),
            "spike_capacity_bytes": self.spikes.capacity() * size_of::<(u32, u32)>(),
        })
    }

    pub(super) fn write_csv(
        &self,
        directory: &Path,
        monitor: &Monitor,
        steps: usize,
        dt: f64,
    ) -> Result<()> {
        let mut trace = BufWriter::new(File::create(directory.join("state.csv"))?);
        let mut spikes = BufWriter::new(File::create(directory.join("spikes.csv"))?);
        writeln!(trace, "tick,t_seconds,i,{}", monitor.variables.join(","))?;
        writeln!(spikes, "tick,t_seconds,i")?;
        let mut sample = 0;
        for tick in 0..steps {
            let time = tick as f64 * dt;
            for &i in &monitor.record {
                write!(trace, "{tick},{time:.17e},{i}")?;
                for _ in &monitor.variables {
                    write!(trace, ",{:.17e}", self.samples[sample])?;
                    sample += 1;
                }
                writeln!(trace)?;
            }
        }
        for &(tick, i) in &self.spikes {
            let time = tick as f64 * dt;
            writeln!(spikes, "{tick},{time:.17e},{i}")?;
        }
        trace.flush()?;
        spikes.flush()?;
        Ok(())
    }
}
