#![feature(lazy_cell)]
#![feature(round_ties_even)]

pub mod cpu;
pub mod instructions;
pub mod io;

use cpu::riscv32::*;
use std::env;
use std::time::Instant;

#[inline]
pub fn convert(compiled: Vec<u32>) -> Vec<u8> {
    let mut output = Vec::new();

    for inst in compiled {
        output.extend(inst.to_le_bytes().to_vec());
    }

    output
}

//#[tokio::main]
fn main() {
    let args: Vec<String> = env::args().collect();

    let threads = num_cpus::get();
    let cores = num_cpus::get_physical();
    init(cores, threads / cores, 1024 * 1024 * 1024);
    #[cfg(feature = "debug")]
    println!("CPU Initialized.");

    let program: Vec<u32> = vec![
        0xff050513, 0x00a00023, 0x00a01223, 0x00a02623, 0x00000583, 0x00401603, 0x00c02683,
        0x00004703, 0x00405783,
    ];

    let instructions: Vec<u32> = program.repeat(1000000);

    println!("instruction count: {}", instructions.len());

    if args.contains(&"--simulator".to_owned()) {
        let mut binary = convert(instructions.clone());

        for arg in args {
            if arg.starts_with("--input") {
                if let Some(file) = arg.split_once('=') {
                    binary = std::fs::read(file.1).unwrap();
                }
            }
        }

        prepare_hart(0, 0, binary.clone());

        let cpu = CPU.read();

        let mut core = cpu.cores[0].lock();
        let mut hart = core.harts[0].lock();

        let now = Instant::now();

        while (hart.pc as usize) < hart.region.1 {
            hart.fetch();

            let mut i: usize = 1;
            while i < hart.pipeline.len() {
                match hart.pipeline[i].stage {
                    Stage::Fetch => unreachable!(),
                    Stage::Decode => hart.decode(),
                    Stage::Execute => hart.execute(),
                    Stage::Memory => hart.memory(),
                    Stage::Write => hart.write(),
                };

                i += 1;
            }
        }

        drop(hart);

        core.stop_hart(0);

        /*
        let mut i: usize = 0;
        while i < hart_count {
            cpu.prepare_hart(i, binary.clone());

            i += 1;
        }

        let mut handles = Vec::with_capacity(hart_count);

        let now = Instant::now();

        for i in 0..1 {
            let h = cpu.harts[i].clone();

            handles.push(thread::spawn(move || {
                let mut hart = h.lock();
                while (hart.pc as usize) < hart.region.1 {
                    let (inst_u32, inst) = hart.fetch();
                    //let oooe = self.oooe.cycle(inst);
                    hart.execute(inst_u32, inst);
                }
                hart.reset();

                #[cfg(feature = "debug")]
                println!("-- HART {} FINISHED --", i);
            }));
        }

        for handle in handles {
            handle.join().unwrap();
        }
        */

        let elapsed = now.elapsed();

        println!(
            "elapsed: {:?}, MIPS: {}",
            elapsed,
            (instructions.len() as f64) / elapsed.as_secs_f64()
        );

        #[cfg(feature = "debug")]
        println!("{}", core.harts[0].lock().get(10)); // print x10 on hart 0
    }
}
