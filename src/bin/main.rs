use boxcars;
use ::ndarray::Axis;
use subtr_actor_spec::*;
use std::env;
use std::fs::File;
use std::io::Write;
use serde_json::json;

fn main() {
    let args: Vec<_> = env::args().collect();
    let data = std::fs::read(&args[1]).unwrap();
    let parsing = boxcars::ParserBuilder::new(&data[..])
        .always_check_crc()
        .must_parse_network_data()
        .parse();
    let replay = parsing.unwrap();

    let mut collector = NDArrayCollector::<f32>::from_strings(
        &["InterpolatedBallRigidBodyNoVelocities"],
        &[
            "InterpolatedPlayerRigidBodyNoVelocities",
            "PlayerBoost",
            "PlayerAnyJump",
            "PlayerDemolishedBy",
        ],
    )
    .unwrap();

    let mut collector2 = NDArrayCollector::<f32>::from_strings(
        &["InterpolatedBallRigidBodyNoVelocities"],
        &[
            "InterpolatedPlayerRigidBodyNoVelocities",
            "PlayerBoost",
            "PlayerAnyJump",
            "PlayerDemolishedBy",
        ],
    )
    .unwrap();

    FrameRateDecorator::new_from_fps(10.0, &mut collector)
    .process_replay(&replay)
    .unwrap();

    let (meta, array) = collector.get_meta_and_ndarray().unwrap();
    
    let result = collector2.process_and_get_meta_and_headers(&replay).unwrap();

    // Extract the shots metadata
    let shots = result.replay_meta.all_headers;   

    // // Write the JSON to a file
    // let output_file = "output.json";
    // let mut file = File::create(output_file).unwrap();
    // file.write_all(serde_json::to_string_pretty(&output_json).unwrap().as_bytes()).unwrap();

    // println!("Data has been written to {}", output_file);


    // Convert the ndarray to a JSON-compatible structure
    let json_array: Vec<Vec<f32>> = array
        .axis_iter(Axis(0))
        .map(|row| row.to_vec())
        .collect();

    let output_json = json!({
        "shots": shots,
        "meta": meta.headers_vec(),
        "array": json_array,
    });

    // Write the JSON to a file
    let output_file = "output.json";
    let mut file = File::create(output_file).unwrap();
    file.write_all(serde_json::to_string_pretty(&output_json).unwrap().as_bytes()).unwrap();

    println!("Array shape is {:?}", array.shape());
    println!("Data has been written to {}", output_file);

}
