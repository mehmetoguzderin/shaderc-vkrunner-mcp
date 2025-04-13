// vkrunner
//
// Copyright 2023 Neil Roberts
//
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice (including the next
// paragraph) shall be included in all copies or substantial portions of the
// Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
// THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
// DEALINGS IN THE SOFTWARE.

//! This module is just for the unit tests. It makes an alternative
//! version of the Stdio, Command and Output structs from std::process
//! that run a fake compiler so we can get code coverage in
//! compiler.rs without depending a real GLSL compiler.

use std::process::Stdio;
use std::io::{Write, BufWriter, BufReader, BufRead};
use std::fs::File;
use std::ffi::OsStr;
use std::num::ParseIntError;

pub struct ExitStatus {
    success: bool,
}

pub struct Output {
    pub status: ExitStatus,
    pub stdout: Vec<u8>,
    pub stderr: Vec<u8>,
}

pub struct Command {
    args: Vec<String>,
}

impl Command {
    pub fn new<S: AsRef<OsStr>>(_program: S) -> Command {
        Command {
            args: Vec::new(),
        }
    }

    pub fn arg<S: AsRef<OsStr>>(&mut self, arg: S) -> &mut Command {
        self.args.push(arg.as_ref().to_str().unwrap().to_string());

        self
    }

    pub fn args<I, S>(&mut self, args: I) -> &mut Command
    where
        I: IntoIterator<Item = S>,
        S: AsRef<OsStr>,
    {
        for arg in args {
            self.arg(arg);
        }

        self
    }

    pub fn stdout<T: Into<Stdio>>(&mut self, _cfg: T) -> &mut Command {
        self
    }

    pub fn stderr<T: Into<Stdio>>(&mut self, _cfg: T) -> &mut Command {
        self
    }

    pub fn output(&mut self) -> std::io::Result<Output> {
        let mut inputs = Vec::new();
        let mut output = None;
        let mut stdout = Vec::new();
        let mut stderr = Vec::new();
        let mut success = true;

        let mut args = self.args.iter();

        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--quiet" => writeln!(&mut stdout, "quiet").unwrap(),
                "-V" => writeln!(&mut stdout, "vulkan_spirv").unwrap(),
                "--target-env" => {
                    writeln!(
                        &mut stdout,
                        "target_env: {}",
                        args.next().unwrap()
                    ).unwrap();
                },
                "-S" => {
                    writeln!(&mut stdout, "stage: {}", args.next().unwrap())
                        .unwrap();
                },
                "-o" => output = Some(args.next().unwrap()),
                other_arg if other_arg.starts_with("-") => {
                    unreachable!("unexpected arg: {}", other_arg);
                },
                input_file => inputs.push(input_file.to_owned()),
            }
        }

        match output {
            None => {
                // Pretend to output the disassembly
                writeln!(&mut stdout, "disassembly").unwrap();
            },
            Some(output) => {
                if let Err(e) = copy_inputs(&inputs, &output) {
                    writeln!(&mut stderr, "{}", e).unwrap();
                    success = false;
                }
            }
        }

        Ok(Output {
            status: ExitStatus { success },
            stdout,
            stderr,
        })
    }
}

impl ExitStatus {
    pub fn success(&self) -> bool {
        self.success
    }
}

fn copy_inputs(inputs: &[String], output: &str) -> Result<(), ParseIntError> {
    let mut output = BufWriter::new(File::create(output).unwrap());

    for input in inputs {
        let input = File::open(input).unwrap();

        for line in BufReader::new(input).lines() {
            for byte in line.unwrap().split_whitespace() {
                let byte_array = [u8::from_str_radix(byte, 16)?];
                output.write_all(&byte_array).unwrap();
            }
        }
    }

    Ok(())
}
