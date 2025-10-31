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
