use mimalloc::MiMalloc;
use snn::Model;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

fn main() {
    let model = Model::grid(100, 100);

    std::fs::write("output.png", model.to_neato_png().unwrap()).unwrap();
}
