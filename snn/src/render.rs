use super::Model;

use graphviz_rust::{
    cmd::{CommandArg, Format},
    dot_structures::{
        Attribute, Edge, EdgeTy, Graph, GraphAttributes, Id, Node, NodeId, Stmt, Vertex,
    },
    exec,
    printer::PrinterContext,
};

pub fn to_neato_png(model: &Model) -> std::io::Result<Vec<u8>> {
    let mut g = Graph::DiGraph {
        id: Id::Plain("model".to_string()),
        strict: false,
        stmts: Vec::new(),
    };

    g.add_stmt(Stmt::GAttribute(GraphAttributes::Graph(vec![
        Attribute(Id::Plain("layout".into()), Id::Plain("neato".into())),
        Attribute(Id::Plain("overlap".into()), Id::Plain("false".into())),
        Attribute(Id::Plain("splines".into()), Id::Plain("line".into())),
        Attribute(Id::Plain("mode".into()), Id::Plain("sgd".into())),
    ])));

    let n = model.v.len();

    for i in 0..n {
        let node_id = NodeId(Id::Plain(format!("n{}", i)), None);
        let node = Node::new(
            node_id.clone(),
            vec![Attribute(
                Id::Plain("shape".into()),
                Id::Plain("point".into()),
            )],
        );
        g.add_stmt(Stmt::Node(node));

        let start = *model.out_offset.get(i).unwrap_or(&0) as usize;
        let end = *model.out_offset.get(i + 1).unwrap_or(&(start as u32)) as usize;

        for eidx in start..end {
            let tgt = model.receiver[eidx] as usize;

            let edge = Edge {
                ty: EdgeTy::Pair(
                    Vertex::N(node_id.clone()),
                    Vertex::N(NodeId(Id::Plain(format!("n{}", tgt)), None)),
                ),
                // no label, no arrowhead
                attributes: vec![Attribute(Id::Plain("dir".into()), Id::Plain("none".into()))],
            };

            g.add_stmt(Stmt::Edge(edge));
        }
    }

    let mut ctx = PrinterContext::default();
    exec(g, &mut ctx, vec![CommandArg::Format(Format::Png)])
}

#[derive(Clone, Copy)]
pub struct NeuronPosition {
    pub center: [f32; 2],
    pub radius: f32,
}

#[derive(Clone, Copy)]
pub struct SynapsePosition {
    pub end1: [f32; 2],
    pub end2: [f32; 2],
}

/// Returns (neuron_positions, synapse_positions)
pub fn layout_graph(model: &Model) -> (Vec<NeuronPosition>, Vec<SynapsePosition>) {
    // 1. Build the same graph as in to_neato_png
    let mut g = Graph::DiGraph {
        id: Id::Plain("model".to_string()),
        strict: false,
        stmts: Vec::new(),
    };

    g.add_stmt(Stmt::GAttribute(GraphAttributes::Graph(vec![
        Attribute(Id::Plain("layout".into()), Id::Plain("neato".into())),
        Attribute(Id::Plain("overlap".into()), Id::Plain("false".into())),
        Attribute(Id::Plain("splines".into()), Id::Plain("line".into())),
        Attribute(Id::Plain("mode".into()), Id::Plain("sgd".into())),
    ])));

    let n = model.v.len();

    for i in 0..n {
        let node_id = NodeId(Id::Plain(format!("n{}", i)), None);
        let node = Node::new(
            node_id.clone(),
            vec![Attribute(
                Id::Plain("shape".into()),
                Id::Plain("point".into()),
            )],
        );
        g.add_stmt(Stmt::Node(node));

        let start = *model.out_offset.get(i).unwrap_or(&0) as usize;
        let end = *model.out_offset.get(i + 1).unwrap_or(&(start as u32)) as usize;

        for eidx in start..end {
            let tgt = model.receiver[eidx] as usize;

            let edge = Edge {
                ty: EdgeTy::Pair(
                    Vertex::N(node_id.clone()),
                    Vertex::N(NodeId(Id::Plain(format!("n{}", tgt)), None)),
                ),
                attributes: vec![Attribute(Id::Plain("dir".into()), Id::Plain("none".into()))],
            };

            g.add_stmt(Stmt::Edge(edge));
        }
    }

    // 2. Ask Graphviz/neato for a plain layout
    let mut ctx = PrinterContext::default();
    let bytes = match exec(g, &mut ctx, vec![CommandArg::Format(Format::Plain)]) {
        Ok(b) => b,
        Err(_) => {
            // Fallback: zeroed neurons and synapses
            let neurons = vec![
                NeuronPosition {
                    center: [0.0, 0.0],
                    radius: 0.0
                };
                n
            ];
            let synapses = model
                .receiver
                .iter()
                .map(|_| SynapsePosition {
                    end1: [0.0, 0.0],
                    end2: [0.0, 0.0],
                })
                .collect();
            return (neurons, synapses);
        }
    };

    let text = String::from_utf8_lossy(&bytes);

    // 3. Parse "node" lines from plain output
    // Format:
    // node <name> <x> <y> <width> <height> <label> ...
    let mut neuron_positions = vec![
        NeuronPosition {
            center: [0.0, 0.0],
            radius: 0.0
        };
        n
    ];

    for line in text.lines() {
        let mut parts = line.split_whitespace();
        if let Some(kind) = parts.next() {
            if kind == "node" {
                if let (Some(name), Some(xs), Some(ys), Some(ws), Some(hs)) = (
                    parts.next(),
                    parts.next(),
                    parts.next(),
                    parts.next(),
                    parts.next(),
                ) {
                    if let Some(idx_str) = name.strip_prefix('n') {
                        if let Ok(idx) = idx_str.parse::<usize>() {
                            if idx < neuron_positions.len() {
                                let x: f32 = xs.parse().unwrap_or(0.0);
                                let y: f32 = ys.parse().unwrap_or(0.0);
                                let w: f32 = ws.parse().unwrap_or(0.0);
                                let h: f32 = hs.parse().unwrap_or(0.0);

                                neuron_positions[idx] = NeuronPosition {
                                    center: [x, y],
                                    radius: (w.max(h)) * 0.5,
                                };
                            }
                        }
                    }
                }
            }
        }
    }

    // 4. Derive synapse positions from neuron positions and the model's adjacency
    // Keeps the original edge order in model.receiver.
    let mut synapse_positions = Vec::with_capacity(model.receiver.len());
    for src in 0..n {
        let start = *model.out_offset.get(src).unwrap_or(&0) as usize;
        let end = *model.out_offset.get(src + 1).unwrap_or(&(start as u32)) as usize;

        let src_pos = neuron_positions[src].center;
        for eidx in start..end {
            let tgt = model.receiver[eidx] as usize;
            let tgt_pos = if tgt < neuron_positions.len() {
                neuron_positions[tgt].center
            } else {
                [0.0, 0.0]
            };

            synapse_positions.push(SynapsePosition {
                end1: src_pos,
                end2: tgt_pos,
            });
        }
    }

    (neuron_positions, synapse_positions)
}
