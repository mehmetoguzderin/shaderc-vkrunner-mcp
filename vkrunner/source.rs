// vkrunner
//
// Copyright (C) 2018 Intel Corporation
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

use std::path::PathBuf;

/// Struct representing the requested source for the data of a
/// [Source]. This can either be a filename to open and read
/// or directly a string containing the source code.
#[derive(Clone, Debug)]
pub(crate) enum Data {
    File { filename: PathBuf },
    String { source: String },
}

#[derive(Clone, Debug)]
/// A token replacement that should be used for the source. The reader
/// should replace any occurences of `token` in the source with the
/// string in `replacement`.
pub(crate) struct TokenReplacement {
    pub token: String,
    pub replacement: String,
}

/// A source for a shader script. The [Source] struct just contains
/// the details of where the data is stored along with any token
/// replacements to be used while reading the data.
#[derive(Clone, Debug)]
pub struct Source {
    token_replacements: Vec<TokenReplacement>,
    data: Data,
}

type TokenReplacementIter<'a> = std::slice::Iter<'a, TokenReplacement>;

impl Source {
    fn from_data(data: Data) -> Source {
        Source {
            token_replacements: Vec::new(),
            data,
        }
    }

    /// Creates a source that will read lines from the given string.
    pub fn from_string(source: String) -> Source {
        Self::from_data(Data::String { source })
    }

    /// Creates a source that will read lines from the given file.
    pub fn from_file(filename: PathBuf) -> Source {
        Self::from_data(Data::File { filename })
    }

    /// Adds a token replacement to the source. When lines are read
    /// from the source, any mentions of the token will be replaced
    /// with the replacement. The replacement can also contain tokens
    /// which will be replaced as well. This can cause the line
    /// reading to fail and return an error if it causes an infinite
    /// loop.
    pub fn add_token_replacement(
        &mut self,
        token: String,
        replacement: String,
    ) {
        self.token_replacements.push(TokenReplacement { token, replacement });
    }

    /// Return an iterator over the token replacements that were
    /// previously set with
    /// [add_token_replacement](Source::add_token_replacement).
    pub(crate) fn token_replacements(&self) -> TokenReplacementIter {
        self.token_replacements.iter()
    }

    /// Get the data that the source points to.
    pub(crate) fn data(&self) -> &Data {
        &self.data
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_constructors() {
        let source = Source::from_string("my script".to_owned());
        assert!(matches!(
            source.data(),
            Data::String { source } if source == "my script"
        ));

        let source = Source::from_file(
            "my_script.shader_test".to_owned().into()
        );
        assert!(matches!(
            source.data(),
            Data::File { filename }
            if filename.to_str().unwrap() == "my_script.shader_test",
        ));
    }

    #[test]
    fn test_token_replacements() {
        let mut source = Source::from_string("test".to_string());

        assert_eq!(source.token_replacements.len(), 0);

        source.add_token_replacement("COLOUR".to_string(), "0xf00".to_string());

        assert_eq!(source.token_replacements.len(), 1);
        assert_eq!(source.token_replacements[0].token, "COLOUR");
        assert_eq!(source.token_replacements[0].replacement, "0xf00");

        let mut iter = source.token_replacements();
        assert_eq!(iter.next().unwrap().token, "COLOUR");
        assert!(iter.next().is_none());

        source.add_token_replacement("X".to_string(), "12".to_string());

        assert_eq!(source.token_replacements.len(), 2);
        assert_eq!(source.token_replacements[0].token, "COLOUR");
        assert_eq!(source.token_replacements[0].replacement, "0xf00");
        assert_eq!(source.token_replacements[1].token, "X");
        assert_eq!(source.token_replacements[1].replacement, "12");

        let mut iter = source.token_replacements();
        assert_eq!(iter.next().unwrap().token, "COLOUR");
        assert_eq!(iter.next().unwrap().token, "X");
        assert!(iter.next().is_none());
    }
}
