// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

use std::error::Error as StdError;
use std::fmt::{Display, Formatter};
use std::str::FromStr;

use datafusion::common::config::ConfigExtension;
use datafusion::common::{config_field, extensions_options};

/// Session-level controls for DataFusion Iceberg scans.
///
/// Each field is tri-state so sessions can defer to table properties unless
/// they intentionally override table-level defaults.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PartitioningOverride {
    /// Defer to the table property, then to the built-in default.
    #[default]
    Auto,
    /// Enable the matching partitioning family for this session.
    Enabled,
    /// Disable the matching partitioning family for this session.
    Disabled,
}

impl Display for PartitioningOverride {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            PartitioningOverride::Auto => f.write_str("auto"),
            PartitioningOverride::Enabled => f.write_str("enabled"),
            PartitioningOverride::Disabled => f.write_str("disabled"),
        }
    }
}

impl FromStr for PartitioningOverride {
    type Err = ParsePartitioningOverrideError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.to_ascii_lowercase().as_str() {
            "auto" => Ok(PartitioningOverride::Auto),
            "enabled" => Ok(PartitioningOverride::Enabled),
            "disabled" => Ok(PartitioningOverride::Disabled),
            _ => Err(ParsePartitioningOverrideError {
                value: value.to_string(),
            }),
        }
    }
}

/// Error returned when parsing a session partitioning override fails.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParsePartitioningOverrideError {
    value: String,
}

impl Display for ParsePartitioningOverrideError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "invalid partitioning override {:?}, expected auto, enabled, or disabled",
            self.value
        )
    }
}

impl StdError for ParsePartitioningOverrideError {}

config_field!(PartitioningOverride);

extensions_options! {
    /// DataFusion Iceberg scan configuration.
    pub struct IcebergScanConfig {
        /// Controls identity partition values as scan output partitioning.
        pub value_partitioning: PartitioningOverride, default = PartitioningOverride::Auto

        /// Controls Iceberg bucket transforms as scan output partitioning.
        pub bucket_execution: PartitioningOverride, default = PartitioningOverride::Auto
    }
}

impl ConfigExtension for IcebergScanConfig {
    const PREFIX: &'static str = "iceberg";
}
