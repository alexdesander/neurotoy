pub mod render;

const DEFAULT_SYNAPSE_WEIGHT: f32 = 0.5;

// SoA
pub struct Model {
    // ----------------------- Leaky Integrate-And-Fire Neurons
    /// Membrane potential v_t
    v: Vec<f32>,
    /// Leak factor alpha = dt / tau, 0 < alpha <= 1
    alpha: f32,
    /// Firing threshold
    v_th: Vec<f32>,
    /// Reset value after spike
    v_reset: Vec<f32>,
    /// Remaining refractory steps; 0 means active
    refrac: Vec<u16>,
    /// Fixed refractory length in steps
    refrac_len: u16,
    /// Spiked flag for each neuron, so we can render neuron charges
    spiked: Vec<bool>,

    // ----------------------- CSR-Synapses
    /// start/end in "receiver" etc for neuron i is out_offset[i]..out_offset[i+1]
    out_offset: Vec<u32>,
    /// Receiver Neuron
    receiver: Vec<u32>,
    /// Synapse weight
    weight: Vec<f32>,
    /// Synaptic fired flag (true means: deliver weight this tick, then clear)
    state: Vec<u32>,
    // ----------------------- Other simulation state
}

impl Default for Model {
    fn default() -> Self {
        Self::empty()
    }
}

impl Model {
    pub fn empty() -> Self {
        // alpha: small leak per step
        // refrac_len: short refractory period
        Self {
            v: Vec::new(),
            alpha: 0.1,
            v_th: Vec::new(),
            v_reset: Vec::new(),
            refrac: Vec::new(),
            refrac_len: 2,
            spiked: Vec::new(),
            out_offset: vec![0],
            receiver: Vec::new(),
            weight: Vec::new(),
            state: Vec::new(),
        }
    }

    /// Build a 1D chain: 0 → 1 → 2 → ... → (n-1).
    /// Each neuron i (except the last) has exactly one outgoing synapse to i+1.
    pub fn line(neurons: usize) -> Self {
        let mut out_offset = Vec::with_capacity(neurons + 1);
        let mut receiver = Vec::with_capacity(neurons.saturating_sub(1));
        let mut weight = Vec::with_capacity(neurons.saturating_sub(1));
        let mut state = Vec::with_capacity(neurons.saturating_sub(1));

        // CSR prefix
        out_offset.push(0);
        for i in 0..neurons {
            if i + 1 < neurons {
                receiver.push((i + 1) as u32);
                weight.push(DEFAULT_SYNAPSE_WEIGHT);
                state.push(0);
            }
            out_offset.push(receiver.len() as u32);
        }

        Self {
            v: vec![0.0; neurons],
            v_th: vec![1.0; neurons],
            v_reset: vec![0.0; neurons],
            refrac: vec![0; neurons],
            spiked: vec![false; neurons],
            out_offset,
            receiver,
            weight,
            state,
            ..Default::default()
        }
    }

    /// Build a 2D grid of size `rows × cols`.
    /// Neuron index is r*cols + c.
    /// Each neuron connects only to its 4 orthogonal neighbours if they exist.
    pub fn grid(rows: usize, cols: usize) -> Self {
        let n = rows.saturating_mul(cols);

        // Preallocate roughly 4 edges per neuron.
        let mut out_offset = Vec::with_capacity(n + 1);
        let mut receiver = Vec::with_capacity(n.saturating_mul(4));
        let mut weight = Vec::with_capacity(n.saturating_mul(4));
        let mut state = Vec::with_capacity(n.saturating_mul(4));

        out_offset.push(0);

        for r in 0..rows {
            for c in 0..cols {
                let idx = r * cols + c;

                // Up
                if r > 0 {
                    let up = (r - 1) * cols + c;
                    receiver.push(up as u32);
                    weight.push(DEFAULT_SYNAPSE_WEIGHT);
                    state.push(0);
                }
                // Down
                if r + 1 < rows {
                    let down = (r + 1) * cols + c;
                    receiver.push(down as u32);
                    weight.push(DEFAULT_SYNAPSE_WEIGHT);
                    state.push(0);
                }
                // Left
                if c > 0 {
                    let left = r * cols + (c - 1);
                    receiver.push(left as u32);
                    weight.push(DEFAULT_SYNAPSE_WEIGHT);
                    state.push(0);
                }
                // Right
                if c + 1 < cols {
                    let right = r * cols + (c + 1);
                    receiver.push(right as u32);
                    weight.push(DEFAULT_SYNAPSE_WEIGHT);
                    state.push(0);
                }

                // Close CSR range for this neuron.
                out_offset.push(receiver.len() as u32);

                debug_assert_eq!(out_offset.len(), idx + 2);
            }
        }

        Self {
            v: vec![0.0; n],
            v_th: vec![1.0; n],
            v_reset: vec![0.0; n],
            refrac: vec![0; n],
            spiked: vec![false; n],
            out_offset,
            receiver,
            weight,
            state,
            ..Default::default()
        }
    }

    pub fn set_charge(&mut self, neuron: u32, charge: f32) {
        self.v[neuron as usize] = charge;
    }

    pub fn get_charge(&self, neuron: u32) -> f32 {
        self.v[neuron as usize]
    }

    /// Simulate the model for one time step.
    ///
    /// Order:
    /// 1) leak membranes
    /// 2) advance refractory counters
    /// 3) deliver all synapses that fired in the previous step
    /// 4) detect spikes and arm synapses for next step
    pub fn tick(&mut self) {
        // 1) Reset spiked neurons
        for i in 0..self.spiked.len() {
            if self.spiked[i] {
                self.v[i] = self.v_reset[i];
                self.refrac[i] = self.refrac_len;
                self.spiked[i] = false;
            } else {
                // Leak membrane potential
                self.v[i] *= 1.0 - self.alpha;
            }
        }

        // 2) Update refrac for neurons
        for r in &mut self.refrac {
            *r = r.saturating_sub(1);
        }

        // 3) Deliver pending synapses (one-tick delay)
        for i in 0..self.state.len() {
            if self.state[i] == 1 {
                let recv = self.receiver[i] as usize;
                self.v[recv] += self.weight[i];
                // clear for next round
                self.state[i] = 0;
            }
        }

        // 4) Update spikes and arm synapses for next tick
        for i in 0..self.v.len() {
            if self.refrac[i] > 0 {
                continue;
            }
            if self.v[i] < self.v_th[i] {
                continue;
            }

            // spike
            self.spiked[i] = true;

            // arm outgoing synapses to fire on the NEXT tick
            let start = self.out_offset[i] as usize;
            let end = self.out_offset[i + 1] as usize;
            for j in start..end {
                self.state[j] = 1;
            }
        }
    }

    pub fn neuron_vs(&self) -> &[f32] {
        &self.v
    }

    pub fn synapse_states(&self) -> &[u32] {
        &self.state
    }

    /// Render this model using Graphviz' **neato** engine and return an SVG in-memory.
    /// Requires the `graphviz-exec` feature and a `dot`/Graphviz installation.
    pub fn to_neato_png(&self) -> std::io::Result<Vec<u8>> {
        render::to_neato_png(self)
    }
}
