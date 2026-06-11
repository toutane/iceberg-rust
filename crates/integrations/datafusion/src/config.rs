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

use datafusion::common::config::ConfigExtension;
use datafusion::common::extensions_options;

extensions_options! {
    /// Iceberg options registered under the `iceberg` DataFusion namespace.
    pub struct IcebergConfig {
        /// Scan planning options.
        pub scan: IcebergScanConfig, default = IcebergScanConfig::default()
    }
}

impl IcebergConfig {
    /// Enable or disable identity-partition based scan hash declarations.
    pub fn with_identity_partitioning_enabled(mut self, enabled: bool) -> Self {
        self.scan.identity_partitioning_enabled = enabled;
        self
    }

    /// Enable or disable Iceberg bucket-transform aware scan planning.
    pub fn with_bucket_execution_enabled(mut self, enabled: bool) -> Self {
        self.scan.bucket_execution_enabled = enabled;
        self
    }
}

impl ConfigExtension for IcebergConfig {
    const PREFIX: &'static str = "iceberg";
}

extensions_options! {
    /// Scan planning options read when `IcebergTableProvider::scan()` builds a plan.
    pub struct IcebergScanConfig {
        /// Enable identity-partition based scan hash declarations.
        pub identity_partitioning_enabled: bool, default = false

        /// Enable pure single-bucket scan planning.
        pub bucket_execution_enabled: bool, default = true
    }
}

#[cfg(test)]
mod tests {
    use datafusion::common::config::ConfigOptions;

    use super::*;

    #[test]
    fn test_iceberg_scan_config_parses_exact_keys() {
        let mut options = ConfigOptions::default();
        options.extensions.insert(IcebergConfig::default());

        let default_config = options.extensions.get::<IcebergConfig>().unwrap();
        assert!(!default_config.scan.identity_partitioning_enabled);
        assert!(default_config.scan.bucket_execution_enabled);

        options
            .set("iceberg.scan.identity_partitioning_enabled", "true")
            .unwrap();
        options
            .set("iceberg.scan.bucket_execution_enabled", "false")
            .unwrap();

        let config = options.extensions.get::<IcebergConfig>().unwrap();
        assert!(config.scan.identity_partitioning_enabled);
        assert!(!config.scan.bucket_execution_enabled);
    }
}
