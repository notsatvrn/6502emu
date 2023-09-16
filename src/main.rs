#![feature(lazy_cell)]
#![feature(round_ties_even)]

pub mod cpu;
pub mod instructions;
pub mod io;

use std::env;
use std::thread;

#[inline]
pub fn convert(compiled: Vec<u32>) -> Vec<u8> {
    let mut output = Vec::new();

    for inst in compiled {
        output.extend(inst.to_le_bytes().to_vec());
    }

    output
}

#[inline]
pub fn hart(i: usize) -> impl Fn() {
    move || {
        let mut harts = cpu::riscv32::HARTS.write();
        let hart = harts.get_mut(i).unwrap();
        while (hart.pc as usize) < hart.region.1 {
            hart.cycle();
        }
        drop(harts);

        println!("-- HART {} FINISHED --", i);
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let hart_count = num_cpus::get();
    cpu::riscv32::init(hart_count, 1024 * 1024);
    println!("CPU Initialized.");

    if args.contains(&"--simulator".to_owned()) {
        let mut binary = convert(vec![
            0xff050513, 0x00a00023, 0x00a01223, 0x00a02623, 0x00000583, 0x00401603, 0x00c02683,
            0x00004703, 0x00405783,
        ]);

        for arg in args {
            if arg.starts_with("--input") {
                if let Some(file) = arg.split_once('=') {
                    binary = std::fs::read(file.1).unwrap();
                }
            }
        }

        let mut i: usize = 0;
        while i < hart_count {
            cpu::riscv32::prepare_hart(i, binary.clone());

            i += 1;
        }

        let handles: Vec<_> = (0..hart_count).map(|i| thread::spawn(hart(i))).collect();

        for handle in handles {
            handle.join().unwrap();
        }

        println!("{}", cpu::riscv32::HARTS.read()[0].x[10]); // print x10 on hart 0
    }
}
