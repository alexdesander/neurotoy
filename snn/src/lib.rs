mod render;

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

    // ----------------------- CSR-Synapses
    /// start/end in "receiver" etc for neuron i is out_offset[i]..out_offset[i+1]
    out_offset: Vec<u32>,
    /// Receiver Neuron
    receiver: Vec<u32>,
    /// Synapse weight
    weight: Vec<f32>,
    /// Synaptic state (exponential PSC)
    state: Vec<f32>,
    /// decay factor beta = exp(-dt / tau_s)
    beta: f32,
    // ----------------------- Other simulation state
}

impl Default for Model {
    fn default() -> Self {
        Self::empty()
    }
}

impl Model {
    pub fn empty() -> Self {
        // Reasonable simulation defaults:
        // alpha: small leak per step
        // beta: slow synaptic decay
        // refrac_len: short refractory period
        Self {
            v: Vec::new(),
            alpha: 0.1,
            v_th: Vec::new(),
            v_reset: Vec::new(),
            refrac: Vec::new(),
            refrac_len: 2,
            out_offset: vec![0],
            receiver: Vec::new(),
            weight: Vec::new(),
            state: Vec::new(),
            beta: 0.9,
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
                weight.push(1.0);
                state.push(0.0);
            }
            out_offset.push(receiver.len() as u32);
        }

        Self {
            v: vec![0.0; neurons],
            v_th: vec![1.0; neurons],
            v_reset: vec![0.0; neurons],
            refrac: vec![0; neurons],
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
                    weight.push(1.0);
                    state.push(0.0);
                }
                // Down
                if r + 1 < rows {
                    let down = (r + 1) * cols + c;
                    receiver.push(down as u32);
                    weight.push(1.0);
                    state.push(0.0);
                }
                // Left
                if c > 0 {
                    let left = r * cols + (c - 1);
                    receiver.push(left as u32);
                    weight.push(1.0);
                    state.push(0.0);
                }
                // Right
                if c + 1 < cols {
                    let right = r * cols + (c + 1);
                    receiver.push(right as u32);
                    weight.push(1.0);
                    state.push(0.0);
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
    pub fn tick(&mut self) {
        // 1) Leak membranes
        for v in &mut self.v {
            *v *= 1.0 - self.alpha;
        }

        // 2) Decay synapses
        for s in &mut self.state {
            *s *= self.beta;
        }

        // 3) Update refrac for neurons
        for r in &mut self.refrac {
            *r = r.saturating_sub(1);
        }

        // 4) Propagate synapse state to neurons
        for (syn, &recv) in self.state.iter().zip(self.receiver.iter()) {
            self.v[recv as usize] += syn;
        }

        // 5) Update spikes
        for i in 0..self.v.len() {
            let v = &mut self.v[i];
            if *v < self.v_th[i] || self.refrac[i] > 0 {
                continue;
            }
            *v = self.v_reset[i];
            self.refrac[i] = self.refrac_len;

            // Send spike (to synapses)
            let start = self.out_offset[i] as usize;
            let end = self.out_offset[i + 1] as usize;
            for j in start..end {
                self.state[j] += self.weight[j];
            }
        }
    }

    /// Render this model using Graphviz' **neato** engine and return an SVG in-memory.
    /// Requires the `graphviz-exec` feature and a `dot`/Graphviz installation.
    pub fn to_neato_png(&self) -> std::io::Result<Vec<u8>> {
        render::to_neato_png(self)
    }
}
