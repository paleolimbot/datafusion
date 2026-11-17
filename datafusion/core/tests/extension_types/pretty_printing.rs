// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use arrow::array::{FixedSizeBinaryArray, Int8Array, RecordBatch};
use arrow_schema::extension::{Bool8, Uuid};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use datafusion::dataframe::DataFrame;
use datafusion::error::Result;
use datafusion::execution::SessionStateBuilder;
use datafusion::prelude::SessionContext;
use datafusion_expr::registry::MemoryExtensionTypeRegistry;
use insta::assert_snapshot;
use std::sync::Arc;

fn test_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![uuid_field()]))
}

fn uuid_field() -> Field {
    Field::new("my_uuids", DataType::FixedSizeBinary(16), false).with_extension_type(Uuid)
}

async fn create_test_table() -> Result<DataFrame> {
    let schema = test_schema();

    // define data.
    let batch = RecordBatch::try_new(
        schema,
        vec![Arc::new(
            FixedSizeBinaryArray::try_from_iter(
                vec![
                    &[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 5, 6],
                ]
                .into_iter(),
            )
            .unwrap(),
        )],
    )?;

    let state = SessionStateBuilder::default()
        .with_extension_type_registry(Arc::new(
            MemoryExtensionTypeRegistry::new_with_canonical_extension_types(),
        ))
        .build();
    let ctx = SessionContext::new_with_state(state);

    ctx.register_batch("test", batch)?;

    ctx.table("test").await
}

#[tokio::test]
async fn test_pretty_print_extension_type_formatter() -> Result<()> {
    let result = create_test_table().await?.to_string().await?;

    assert_snapshot!(
        result,
        @r"
    +--------------------------------------+
    | my_uuids                             |
    +--------------------------------------+
    | 00000000-0000-0000-0000-000000000000 |
    | 00010203-0405-0607-0809-000102030506 |
    +--------------------------------------+
    "
    );

    Ok(())
}

#[tokio::test]
async fn test_bool8_cast_uses_registered_cast_extension() -> Result<()> {
    let field = Field::new("value", DataType::Int8, true).with_extension_type(Bool8);
    let batch = RecordBatch::try_new(
        Arc::new(Schema::new(vec![field])),
        vec![Arc::new(Int8Array::from(vec![
            Some(0),
            Some(1),
            Some(-1),
            None,
        ]))],
    )?;
    let state = SessionStateBuilder::default()
        .with_extension_type_registry(Arc::new(
            MemoryExtensionTypeRegistry::new_with_canonical_extension_types(),
        ))
        .build();
    let ctx = SessionContext::new_with_state(state);
    ctx.register_batch("bool8_values", batch)?;

    let result = ctx
        .sql("SELECT CAST(value AS VARCHAR) AS value FROM bool8_values")
        .await?
        .to_string()
        .await?;
    assert_snapshot!(
        result,
        @r"
    +-------+
    | value |
    +-------+
    | false |
    | true  |
    | true  |
    |       |
    +-------+
    "
    );
    Ok(())
}
