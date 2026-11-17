// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! FunctionRegistry trait

use crate::expr_rewriter::FunctionRewrite;
use crate::higher_order_function::HigherOrderUDF;
use crate::planner::ExprPlanner;
use crate::{AggregateUDF, ScalarUDF, UserDefinedLogicalNode, WindowUDF};
use arrow::array::{ArrayRef, Int8Array, StringArray};
use arrow::compute::cast_with_options;
use arrow::datatypes::Field;
use arrow_schema::DataType;
use arrow_schema::extension::{
    Bool8, ExtensionType, FixedShapeTensor, Json, Opaque, TimestampWithOffset, Uuid,
    VariableShapeTensor,
};
use datafusion_common::metadata::format_type_and_metadata;
use datafusion_common::nested_struct::{CastExtension, DFCastOptions};
use datafusion_common::types::{
    DFBool8, DFExtensionTypeRef, DFFixedShapeTensor, DFJson, DFOpaque,
    DFTimestampWithOffset, DFUuid, DFVariableShapeTensor,
};
use datafusion_common::{
    HashMap, Result, internal_err, not_impl_err, plan_datafusion_err,
};
use std::collections::HashSet;
use std::fmt::{Debug, Formatter};
use std::sync::{Arc, RwLock};

/// A registry knows how to build logical expressions out of user-defined function' names
pub trait FunctionRegistry {
    /// Returns names of all available scalar user defined functions.
    fn udfs(&self) -> HashSet<String>;

    /// Returns names of all available higher order user defined functions.
    fn higher_order_function_names(&self) -> HashSet<String>;

    /// Returns names of all available aggregate user defined functions.
    fn udafs(&self) -> HashSet<String>;

    /// Returns names of all available window user defined functions.
    fn udwfs(&self) -> HashSet<String>;

    /// Returns a reference to the user defined scalar function (udf) named
    /// `name`.
    fn udf(&self, name: &str) -> Result<Arc<ScalarUDF>>;

    /// Returns a reference to the user defined higher order function named
    /// `name`.
    fn higher_order_function(&self, name: &str) -> Result<Arc<HigherOrderUDF>>;

    /// Returns a reference to the user defined aggregate function (udaf) named
    /// `name`.
    fn udaf(&self, name: &str) -> Result<Arc<AggregateUDF>>;

    /// Returns a reference to the user defined window function (udwf) named
    /// `name`.
    fn udwf(&self, name: &str) -> Result<Arc<WindowUDF>>;

    /// Registers a new [`ScalarUDF`], returning any previously registered
    /// implementation.
    ///
    /// Returns an error (the default) if the function can not be registered,
    /// for example if the registry is read only.
    fn register_udf(&mut self, _udf: Arc<ScalarUDF>) -> Result<Option<Arc<ScalarUDF>>> {
        not_impl_err!("Registering ScalarUDF")
    }
    /// Registers a new [`HigherOrderUDF`], returning any previously registered
    /// implementation.
    ///
    /// Returns an error (the default) if the function can not be registered,
    /// for example if the registry is read only.
    fn register_higher_order_function(
        &mut self,
        _function: Arc<HigherOrderUDF>,
    ) -> Result<Option<Arc<HigherOrderUDF>>> {
        not_impl_err!("Registering HigherOrderUDF")
    }
    /// Registers a new [`AggregateUDF`], returning any previously registered
    /// implementation.
    ///
    /// Returns an error (the default) if the function can not be registered,
    /// for example if the registry is read only.
    fn register_udaf(
        &mut self,
        _udaf: Arc<AggregateUDF>,
    ) -> Result<Option<Arc<AggregateUDF>>> {
        not_impl_err!("Registering AggregateUDF")
    }
    /// Registers a new [`WindowUDF`], returning any previously registered
    /// implementation.
    ///
    /// Returns an error (the default) if the function can not be registered,
    /// for example if the registry is read only.
    fn register_udwf(&mut self, _udaf: Arc<WindowUDF>) -> Result<Option<Arc<WindowUDF>>> {
        not_impl_err!("Registering WindowUDF")
    }

    /// Deregisters a [`ScalarUDF`], returning the implementation that was
    /// deregistered.
    ///
    /// Returns an error (the default) if the function can not be deregistered,
    /// for example if the registry is read only.
    fn deregister_udf(&mut self, _name: &str) -> Result<Option<Arc<ScalarUDF>>> {
        not_impl_err!("Deregistering ScalarUDF")
    }

    /// Deregisters a [`HigherOrderUDF`], returning the implementation that was
    /// deregistered.
    ///
    /// Returns an error (the default) if the function can not be deregistered,
    /// for example if the registry is read only.
    fn deregister_higher_order_function(
        &mut self,
        _name: &str,
    ) -> Result<Option<Arc<HigherOrderUDF>>> {
        not_impl_err!("Deregistering HigherOrderUDF")
    }

    /// Deregisters a [`AggregateUDF`], returning the implementation that was
    /// deregistered.
    ///
    /// Returns an error (the default) if the function can not be deregistered,
    /// for example if the registry is read only.
    fn deregister_udaf(&mut self, _name: &str) -> Result<Option<Arc<AggregateUDF>>> {
        not_impl_err!("Deregistering AggregateUDF")
    }

    /// Deregisters a [`WindowUDF`], returning the implementation that was
    /// deregistered.
    ///
    /// Returns an error (the default) if the function can not be deregistered,
    /// for example if the registry is read only.
    fn deregister_udwf(&mut self, _name: &str) -> Result<Option<Arc<WindowUDF>>> {
        not_impl_err!("Deregistering WindowUDF")
    }

    /// Registers a new [`FunctionRewrite`] with the registry.
    ///
    /// `FunctionRewrite` rules are used to rewrite certain / operators in the
    /// logical plan to function calls.  For example `a || b` might be written to
    /// `array_concat(a, b)`.
    ///
    /// This allows the behavior of operators to be customized by the user.
    fn register_function_rewrite(
        &mut self,
        _rewrite: Arc<dyn FunctionRewrite + Send + Sync>,
    ) -> Result<()> {
        not_impl_err!("Registering FunctionRewrite")
    }

    /// Set of all registered [`ExprPlanner`]s
    fn expr_planners(&self) -> Vec<Arc<dyn ExprPlanner>>;

    /// Registers a new [`ExprPlanner`] with the registry.
    fn register_expr_planner(
        &mut self,
        _expr_planner: Arc<dyn ExprPlanner>,
    ) -> Result<()> {
        not_impl_err!("Registering ExprPlanner")
    }
}

/// Serializer and deserializer registry for extensions like [UserDefinedLogicalNode].
pub trait SerializerRegistry: Debug + Send + Sync {
    /// Serialize this node to a byte array. This serialization should not include
    /// input plans.
    fn serialize_logical_plan(
        &self,
        node: &dyn UserDefinedLogicalNode,
    ) -> Result<Vec<u8>>;

    /// Deserialize user defined logical plan node ([UserDefinedLogicalNode]) from
    /// bytes.
    fn deserialize_logical_plan(
        &self,
        name: &str,
        bytes: &[u8],
    ) -> Result<Arc<dyn UserDefinedLogicalNode>>;
}

/// A  [`FunctionRegistry`] that uses in memory [`HashMap`]s
#[derive(Default, Debug)]
pub struct MemoryFunctionRegistry {
    /// Scalar Functions
    udfs: HashMap<String, Arc<ScalarUDF>>,
    /// Aggregate Functions
    udafs: HashMap<String, Arc<AggregateUDF>>,
    /// Window Functions
    udwfs: HashMap<String, Arc<WindowUDF>>,
    /// Higher Order Functions
    higher_order_functions: HashMap<String, Arc<HigherOrderUDF>>,
}

impl MemoryFunctionRegistry {
    pub fn new() -> Self {
        Self::default()
    }
}

impl FunctionRegistry for MemoryFunctionRegistry {
    fn udfs(&self) -> HashSet<String> {
        self.udfs.keys().cloned().collect()
    }

    fn udf(&self, name: &str) -> Result<Arc<ScalarUDF>> {
        self.udfs
            .get(name)
            .cloned()
            .ok_or_else(|| plan_datafusion_err!("Function {name} not found"))
    }

    fn higher_order_function(&self, name: &str) -> Result<Arc<HigherOrderUDF>> {
        self.higher_order_functions
            .get(name)
            .cloned()
            .ok_or_else(|| plan_datafusion_err!("Higher Order Function {name} not found"))
    }

    fn udaf(&self, name: &str) -> Result<Arc<AggregateUDF>> {
        self.udafs
            .get(name)
            .cloned()
            .ok_or_else(|| plan_datafusion_err!("Aggregate Function {name} not found"))
    }

    fn udwf(&self, name: &str) -> Result<Arc<WindowUDF>> {
        self.udwfs
            .get(name)
            .cloned()
            .ok_or_else(|| plan_datafusion_err!("Window Function {name} not found"))
    }

    fn register_udf(&mut self, udf: Arc<ScalarUDF>) -> Result<Option<Arc<ScalarUDF>>> {
        Ok(self.udfs.insert(udf.name().to_string(), udf))
    }
    fn register_higher_order_function(
        &mut self,
        function: Arc<HigherOrderUDF>,
    ) -> Result<Option<Arc<HigherOrderUDF>>> {
        Ok(self
            .higher_order_functions
            .insert(function.name().into(), function))
    }
    fn register_udaf(
        &mut self,
        udaf: Arc<AggregateUDF>,
    ) -> Result<Option<Arc<AggregateUDF>>> {
        Ok(self.udafs.insert(udaf.name().into(), udaf))
    }
    fn register_udwf(&mut self, udaf: Arc<WindowUDF>) -> Result<Option<Arc<WindowUDF>>> {
        Ok(self.udwfs.insert(udaf.name().into(), udaf))
    }

    fn expr_planners(&self) -> Vec<Arc<dyn ExprPlanner>> {
        vec![]
    }

    fn higher_order_function_names(&self) -> HashSet<String> {
        self.higher_order_functions.keys().cloned().collect()
    }

    fn udafs(&self) -> HashSet<String> {
        self.udafs.keys().cloned().collect()
    }

    fn udwfs(&self) -> HashSet<String> {
        self.udwfs.keys().cloned().collect()
    }
}

/// A cheaply cloneable pointer to an [ExtensionTypeRegistry].
pub type ExtensionTypeRegistryRef = Arc<dyn ExtensionTypeRegistry>;

/// Manages [`ExtensionTypeRegistration`]s, which allow users to register custom behavior for
/// extension types.
///
/// Each registration is connected to the extension type name, which can also be looked up to get
/// the registration.
pub trait ExtensionTypeRegistry: Debug + Send + Sync {
    /// Returns a reference to registration of an extension type named `name`.
    ///
    /// Returns an error if there is no extension type with that name.
    fn extension_type_registration(
        &self,
        name: &str,
    ) -> Result<ExtensionTypeRegistrationRef>;

    /// Creates a [`DFExtensionTypeRef`] from the type information in the `field`.
    ///
    /// The result `Ok(None)` indicates that there is no extension type metadata. Returns an error
    /// if the extension type in the metadata is not found.
    fn create_extension_type_for_field(
        &self,
        field: &Field,
    ) -> Result<Option<DFExtensionTypeRef>> {
        let Some(extension_type_name) = field.extension_type_name() else {
            return Ok(None);
        };

        let registration = self.extension_type_registration(extension_type_name)?;
        registration
            .create_df_extension_type(field.data_type(), field.extension_type_metadata())
            .map(Some)
    }

    /// Returns all registered [ExtensionTypeRegistration].
    fn extension_type_registrations(&self) -> Vec<ExtensionTypeRegistrationRef>;

    /// Registers a new [ExtensionTypeRegistrationRef], returning any previously registered
    /// implementation.
    ///
    /// Returns an error if the type cannot be registered, for example, if the registry is
    /// read-only.
    fn add_extension_type_registration(
        &self,
        extension_type: ExtensionTypeRegistrationRef,
    ) -> Result<Option<ExtensionTypeRegistrationRef>>;

    /// Extends the registry with the provided extension types.
    ///
    /// Returns an error if the type cannot be registered, for example, if the registry is
    /// read-only.
    fn extend(&self, extension_types: &[ExtensionTypeRegistrationRef]) -> Result<()> {
        for extension_type in extension_types.iter().cloned() {
            self.add_extension_type_registration(extension_type)?;
        }
        Ok(())
    }

    /// Deregisters an extension type registration with the name `name`, returning the
    /// implementation that was deregistered.
    ///
    /// Returns an error if the type cannot be deregistered, for example, if the registry is
    /// read-only.
    fn remove_extension_type_registration(
        &self,
        name: &str,
    ) -> Result<Option<ExtensionTypeRegistrationRef>>;

    /// Returns a [`CastExtension`] that may be used to override casts to and from extension types
    /// in the registry.
    fn cast_extension(&self) -> Option<Arc<dyn CastExtension>> {
        None
    }

    /// Adds a cast extension that may override casts between arbitrary fields.
    fn add_cast_extension(&self, _extension: Arc<dyn CastExtension>) -> Result<()> {
        not_impl_err!(
            "ExtensionTypeRegistry does not support registration of custom cast extensions"
        )
    }
}

/// A factory that creates instances of extension types from a storage [`DataType`] and the
/// metadata.
pub type ExtensionTypeFactory =
    dyn Fn(&DataType, Option<&str>) -> Result<DFExtensionTypeRef> + Send + Sync;

/// A cheaply cloneable pointer to an [ExtensionTypeRegistration].
pub type ExtensionTypeRegistrationRef = Arc<ExtensionTypeRegistration>;

/// The registration of an extension type. Implementations of this trait are responsible for
/// *creating* instances of [`DFExtensionType`] that represent the entire semantics of an extension
/// type.
///
/// # Why do we need a Registration?
///
/// A good question is why this trait is even necessary. Why not directly register the
/// [`DFExtensionType`] in a registry?
///
/// While this works for extension types requiring no additional metadata (e.g., `arrow.uuid`), it
/// does not work for more complex extension types with metadata. For example, consider an extension
/// type `custom.shortened(n)` that aims to short the pretty-printing string to `n` characters.
/// Here, `n` is a parameter of the extension type and should be a field in the struct that
/// implements the [`DFExtensionType`]. The job of the registration is to read the metadata from the
/// field and create the corresponding [`DFExtensionType`] instance with the correct `n` set.
///
/// [`DFExtensionType`]: datafusion_common::types::DFExtensionType
pub struct ExtensionTypeRegistration {
    /// The name of the extension type.
    name: String,
    /// A function that creates an instance of [`DFExtensionTypeRef`] from the storage type and the
    /// metadata.
    factory: Box<ExtensionTypeFactory>,
}

impl ExtensionTypeRegistration {
    /// Creates a new registration for an extension type. The factory is required to validate that
    /// the storage [`DataType`] is compatible with the extension type.
    pub fn new_arc(
        name: impl Into<String>,
        factory: impl Fn(&DataType, Option<&str>) -> Result<DFExtensionTypeRef>
        + Send
        + Sync
        + 'static,
    ) -> ExtensionTypeRegistrationRef {
        Arc::new(Self {
            name: name.into(),
            factory: Box::new(factory),
        })
    }
}

impl ExtensionTypeRegistration {
    /// The name of the extension type.
    ///
    /// This name will be used to find the correct [ExtensionTypeRegistration] when an extension
    /// type is encountered.
    pub fn type_name(&self) -> &str {
        &self.name
    }

    /// Creates an extension type instance from the optional metadata. The name of the extension
    /// type is not a parameter as it's already defined by the registration itself.
    pub fn create_df_extension_type(
        &self,
        storage_type: &DataType,
        metadata: Option<&str>,
    ) -> Result<DFExtensionTypeRef> {
        self.factory.as_ref()(storage_type, metadata)
    }
}

impl Debug for ExtensionTypeRegistration {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DefaultExtensionTypeRegistration")
            .field("type_name", &self.name)
            .finish()
    }
}

/// An [`ExtensionTypeRegistry`] that uses in memory [`HashMap`]s.
#[derive(Clone, Debug)]
pub struct MemoryExtensionTypeRegistry {
    /// Holds a mapping between the name of an extension type and its logical type.
    extension_types: Arc<RwLock<HashMap<String, ExtensionTypeRegistrationRef>>>,
    /// Holds a list of custom cast rules
    cast_extensions: Arc<RwLock<Vec<Arc<dyn CastExtension>>>>,
}

impl Default for MemoryExtensionTypeRegistry {
    fn default() -> Self {
        Self::new_empty()
    }
}

impl MemoryExtensionTypeRegistry {
    /// Creates an empty [MemoryExtensionTypeRegistry].
    pub fn new_empty() -> Self {
        Self {
            extension_types: Arc::new(RwLock::new(HashMap::new())),
            cast_extensions: Arc::new(RwLock::new(Vec::new())),
        }
    }

    /// Pre-registers the [canonical extension types](https://arrow.apache.org/docs/format/CanonicalExtensions.html)
    /// in the extension type registry.
    pub fn new_with_canonical_extension_types() -> Self {
        let mapping = [
            ExtensionTypeRegistration::new_arc(
                FixedShapeTensor::NAME,
                |storage_type, metadata| {
                    Ok(Arc::new(DFFixedShapeTensor::try_new(
                        storage_type,
                        FixedShapeTensor::deserialize_metadata(metadata)?,
                    )?))
                },
            ),
            ExtensionTypeRegistration::new_arc(
                VariableShapeTensor::NAME,
                |storage_type, metadata| {
                    Ok(Arc::new(DFVariableShapeTensor::try_new(
                        storage_type,
                        VariableShapeTensor::deserialize_metadata(metadata)?,
                    )?))
                },
            ),
            ExtensionTypeRegistration::new_arc(Json::NAME, |storage_type, metadata| {
                Ok(Arc::new(DFJson::try_new(
                    storage_type,
                    Json::deserialize_metadata(metadata)?,
                )?))
            }),
            ExtensionTypeRegistration::new_arc(Uuid::NAME, |storage_type, metadata| {
                Ok(Arc::new(DFUuid::try_new(
                    storage_type,
                    Uuid::deserialize_metadata(metadata)?,
                )?))
            }),
            ExtensionTypeRegistration::new_arc(Opaque::NAME, |storage_type, metadata| {
                Ok(Arc::new(DFOpaque::try_new(
                    storage_type,
                    Opaque::deserialize_metadata(metadata)?,
                )?))
            }),
            ExtensionTypeRegistration::new_arc(Bool8::NAME, |storage_type, metadata| {
                Ok(Arc::new(DFBool8::try_new(
                    storage_type,
                    Bool8::deserialize_metadata(metadata)?,
                )?))
            }),
            ExtensionTypeRegistration::new_arc(
                TimestampWithOffset::NAME,
                |storage_type, metadata| {
                    Ok(Arc::new(DFTimestampWithOffset::try_new(
                        storage_type,
                        TimestampWithOffset::deserialize_metadata(metadata)?,
                    )?))
                },
            ),
        ];

        let mut extension_types = HashMap::new();
        for registration in mapping.into_iter() {
            extension_types.insert(registration.type_name().to_owned(), registration);
        }

        let registry = Self {
            extension_types: Arc::new(RwLock::new(HashMap::from(extension_types))),
            cast_extensions: Arc::new(RwLock::new(Vec::new())),
        };
        let casts: Arc<dyn CastExtension> = Arc::new(Bool8Casts);
        registry
            .add_cast_extension(casts)
            .expect("in-memory cast registry is writable");
        registry
    }

    /// Creates a new [MemoryExtensionTypeRegistry] with the provided `types`.
    ///
    /// # Errors
    ///
    /// Returns an error if one of the `types` is a native type.
    pub fn new_with_types(
        types: impl IntoIterator<Item = ExtensionTypeRegistrationRef>,
    ) -> Result<Self> {
        let extension_types = types
            .into_iter()
            .map(|t| (t.type_name().to_owned(), t))
            .collect::<HashMap<_, _>>();
        Ok(Self {
            extension_types: Arc::new(RwLock::new(extension_types)),
            cast_extensions: Arc::new(RwLock::new(Vec::new())),
        })
    }

    /// Returns a list of all registered types.
    pub fn all_extension_types(&self) -> Vec<ExtensionTypeRegistrationRef> {
        self.extension_types
            .read()
            .expect("Extension type registry lock poisoned")
            .values()
            .cloned()
            .collect()
    }
}

impl ExtensionTypeRegistry for MemoryExtensionTypeRegistry {
    fn extension_type_registration(
        &self,
        name: &str,
    ) -> Result<ExtensionTypeRegistrationRef> {
        self.extension_types
            .write()
            .expect("Extension type registry lock poisoned")
            .get(name)
            .ok_or_else(|| plan_datafusion_err!("Logical type not found."))
            .cloned()
    }

    fn extension_type_registrations(&self) -> Vec<ExtensionTypeRegistrationRef> {
        self.extension_types
            .read()
            .expect("Extension type registry lock poisoned")
            .values()
            .cloned()
            .collect()
    }

    fn add_extension_type_registration(
        &self,
        extension_type: ExtensionTypeRegistrationRef,
    ) -> Result<Option<ExtensionTypeRegistrationRef>> {
        Ok(self
            .extension_types
            .write()
            .expect("Extension type registry lock poisoned")
            .insert(extension_type.type_name().to_owned(), extension_type))
    }

    fn remove_extension_type_registration(
        &self,
        name: &str,
    ) -> Result<Option<ExtensionTypeRegistrationRef>> {
        Ok(self
            .extension_types
            .write()
            .expect("Extension type registry lock poisoned")
            .remove(name))
    }

    fn cast_extension(&self) -> Option<Arc<dyn CastExtension>> {
        (!self
            .cast_extensions
            .read()
            .expect("Cast extension registry lock poisoned")
            .is_empty())
        .then(|| Arc::new(self.clone()) as Arc<dyn CastExtension>)
    }

    fn add_cast_extension(&self, extension: Arc<dyn CastExtension>) -> Result<()> {
        self.cast_extensions
            .write()
            .expect("Cast extension registry lock poisoned")
            .push(extension);
        Ok(())
    }
}

impl From<HashMap<String, ExtensionTypeRegistrationRef>> for MemoryExtensionTypeRegistry {
    fn from(value: HashMap<String, ExtensionTypeRegistrationRef>) -> Self {
        Self {
            extension_types: Arc::new(RwLock::new(value)),
            cast_extensions: Arc::new(RwLock::new(Vec::new())),
        }
    }
}

impl MemoryExtensionTypeRegistry {
    fn matching_cast_extension(
        &self,
        source_field: &Field,
        target_field: &Field,
        options: &DFCastOptions<'_>,
    ) -> Result<Option<Arc<dyn CastExtension>>> {
        let registrations = self
            .cast_extensions
            .read()
            .expect("Cast extension registry lock poisoned")
            .clone();

        let mut matching_indices = Vec::new();
        for (index, registration) in registrations.iter().enumerate() {
            if registration.can_cast_fields(source_field, target_field, options)? {
                matching_indices.push(index);
            }
        }

        match matching_indices.as_slice() {
            [] => Ok(None),
            [index] => Ok(Some(Arc::clone(&registrations[*index]))),
            _ => internal_err!(
                "Multiple registered cast extensions can cast field from {} to {}",
                format_type_and_metadata(
                    source_field.data_type(),
                    Some(source_field.metadata())
                ),
                format_type_and_metadata(
                    target_field.data_type(),
                    Some(target_field.metadata())
                )
            ),
        }
    }
}

impl CastExtension for MemoryExtensionTypeRegistry {
    fn can_cast_fields(
        &self,
        source_field: &Field,
        target_field: &Field,
        options: &DFCastOptions<'_>,
    ) -> Result<bool> {
        Ok(self
            .matching_cast_extension(source_field, target_field, options)?
            .is_some())
    }

    fn is_injective_fields(
        &self,
        source_field: &Field,
        target_field: &Field,
        options: &DFCastOptions<'_>,
    ) -> Result<bool> {
        let Some(extension) =
            self.matching_cast_extension(source_field, target_field, options)?
        else {
            return Ok(false);
        };
        extension.is_injective_fields(source_field, target_field, options)
    }

    fn is_order_preserving_fields(
        &self,
        source_field: &Field,
        target_field: &Field,
        options: &DFCastOptions<'_>,
    ) -> Result<bool> {
        let Some(extension) =
            self.matching_cast_extension(source_field, target_field, options)?
        else {
            return Ok(false);
        };
        extension.is_order_preserving_fields(source_field, target_field, options)
    }

    fn preserves_nulls_fields(
        &self,
        source_field: &Field,
        target_field: &Field,
        options: &DFCastOptions<'_>,
    ) -> Result<bool> {
        let Some(extension) =
            self.matching_cast_extension(source_field, target_field, options)?
        else {
            return Ok(false);
        };
        extension.preserves_nulls_fields(source_field, target_field, options)
    }

    fn cast_array_fields(
        &self,
        array: &ArrayRef,
        source_field: &Field,
        target_field: &Field,
        options: &DFCastOptions<'_>,
    ) -> Result<ArrayRef> {
        let Some(extension) =
            self.matching_cast_extension(source_field, target_field, options)?
        else {
            return internal_err!(
                "No registered cast extension can cast field from {} to {}",
                source_field.data_type(),
                target_field.data_type()
            );
        };
        extension.cast_array_fields(array, source_field, target_field, options)
    }
}

#[derive(Debug)]
struct Bool8Casts;

impl CastExtension for Bool8Casts {
    fn can_cast_fields(
        &self,
        source_field: &Field,
        target_field: &Field,
        _options: &DFCastOptions<'_>,
    ) -> Result<bool> {
        let source_is_bool8 = validate_bool8_field(source_field)?;
        let target_is_bool8 = validate_bool8_field(target_field)?;

        Ok(match (source_is_bool8, target_is_bool8) {
            (true, true) => true,
            (true, false) => {
                target_field.extension_type_name().is_none()
                    && (target_field.data_type() == &DataType::Int8
                        || is_string_type(target_field.data_type()))
            }
            (false, true) => {
                source_field.extension_type_name().is_none()
                    && source_field.data_type() == &DataType::Int8
            }
            (false, false) => false,
        })
    }

    fn is_injective_fields(
        &self,
        source_field: &Field,
        target_field: &Field,
        _options: &DFCastOptions<'_>,
    ) -> Result<bool> {
        Ok(!is_string_type(target_field.data_type())
            && (source_field.extension_type_name() == Some(Bool8::NAME)
                || target_field.extension_type_name() == Some(Bool8::NAME)))
    }

    fn is_order_preserving_fields(
        &self,
        source_field: &Field,
        target_field: &Field,
        options: &DFCastOptions<'_>,
    ) -> Result<bool> {
        self.is_injective_fields(source_field, target_field, options)
    }

    fn preserves_nulls_fields(
        &self,
        _source_field: &Field,
        _target_field: &Field,
        _options: &DFCastOptions<'_>,
    ) -> Result<bool> {
        Ok(true)
    }

    fn cast_array_fields(
        &self,
        array: &ArrayRef,
        source_field: &Field,
        target_field: &Field,
        options: &DFCastOptions<'_>,
    ) -> Result<ArrayRef> {
        if !self.can_cast_fields(source_field, target_field, options)? {
            return internal_err!("Bool8 cast cannot cast supplied fields");
        }
        if array.data_type() != source_field.data_type() {
            return internal_err!(
                "Bool8 cast received array of type {}, expected {}",
                array.data_type(),
                source_field.data_type()
            );
        }

        if source_field.extension_type_name() == Some(Bool8::NAME)
            && is_string_type(target_field.data_type())
        {
            let Some(array) = array.as_any().downcast_ref::<Int8Array>() else {
                return internal_err!("Bool8 storage array must have Int8 array type");
            };
            let strings: ArrayRef =
                Arc::new(StringArray::from_iter(array.iter().map(|value| {
                    value.map(|value| if value == 0 { "false" } else { "true" })
                })));
            return Ok(cast_with_options(
                &strings,
                target_field.data_type(),
                &options.arrow_options,
            )?);
        }

        Ok(Arc::clone(array))
    }
}

fn validate_bool8_field(field: &Field) -> Result<bool> {
    if field.extension_type_name() != Some(Bool8::NAME) {
        return Ok(false);
    }
    DFBool8::try_new(
        field.data_type(),
        Bool8::deserialize_metadata(field.extension_type_metadata())?,
    )?;
    Ok(true)
}

fn is_string_type(data_type: &DataType) -> bool {
    matches!(
        data_type,
        DataType::Utf8 | DataType::LargeUtf8 | DataType::Utf8View
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{AsArray, LargeStringArray, StringViewArray};

    fn bool8_field(data_type: DataType) -> Field {
        Field::new("value", data_type, true).with_extension_type(Bool8)
    }

    fn options(extension: &dyn CastExtension) -> DFCastOptions<'_> {
        DFCastOptions::default().with_cast_extension(extension)
    }

    #[test]
    fn empty_registry_has_no_cast_extension() {
        assert!(
            MemoryExtensionTypeRegistry::new_empty()
                .cast_extension()
                .is_none()
        );
    }

    #[test]
    fn bool8_casts_to_string_types() -> Result<()> {
        let registry = MemoryExtensionTypeRegistry::new_with_canonical_extension_types();
        let extension = registry.cast_extension().unwrap();
        let options = options(extension.as_ref());
        let source_field = bool8_field(DataType::Int8);
        let values: ArrayRef =
            Arc::new(Int8Array::from(vec![Some(0), Some(1), Some(-1), None]));

        let utf8 = extension.cast_array_fields(
            &values,
            &source_field,
            &Field::new("value", DataType::Utf8, true),
            &options,
        )?;
        assert_eq!(
            utf8.as_string::<i32>().iter().collect::<Vec<_>>(),
            vec![Some("false"), Some("true"), Some("true"), None]
        );

        let large_utf8 = extension.cast_array_fields(
            &values,
            &source_field,
            &Field::new("value", DataType::LargeUtf8, true),
            &options,
        )?;
        assert!(large_utf8.as_any().is::<LargeStringArray>());

        let utf8_view = extension.cast_array_fields(
            &values,
            &source_field,
            &Field::new("value", DataType::Utf8View, true),
            &options,
        )?;
        assert!(utf8_view.as_any().is::<StringViewArray>());
        Ok(())
    }

    #[test]
    fn bool8_casts_to_and_from_storage_type() -> Result<()> {
        let registry = MemoryExtensionTypeRegistry::new_with_canonical_extension_types();
        let extension = registry.cast_extension().unwrap();
        let options = options(extension.as_ref());
        let bool8 = bool8_field(DataType::Int8);
        let storage = Field::new("value", DataType::Int8, true);
        let values: ArrayRef = Arc::new(Int8Array::from(vec![0, 1, -1]));

        let to_storage =
            extension.cast_array_fields(&values, &bool8, &storage, &options)?;
        let from_storage =
            extension.cast_array_fields(&values, &storage, &bool8, &options)?;
        assert!(Arc::ptr_eq(&values, &to_storage));
        assert!(Arc::ptr_eq(&values, &from_storage));
        Ok(())
    }

    #[test]
    fn bool8_cast_rejects_invalid_storage_type() {
        let registry = MemoryExtensionTypeRegistry::new_with_canonical_extension_types();
        let extension = registry.cast_extension().unwrap();
        let options = options(extension.as_ref());
        let invalid_bool8 = Field::new("value", DataType::Int16, true)
            .with_metadata([("ARROW:extension:name".to_owned(), Bool8::NAME.to_owned())]);
        let string = Field::new("value", DataType::Utf8, true);

        assert!(
            extension
                .can_cast_fields(&invalid_bool8, &string, &options)
                .is_err()
        );
        assert!(
            extension
                .can_cast_fields(&string, &invalid_bool8, &options)
                .is_err()
        );
    }

    #[test]
    fn cast_registry_rejects_ambiguous_matches() -> Result<()> {
        let registry = MemoryExtensionTypeRegistry::new_with_canonical_extension_types();
        let bool8 = bool8_field(DataType::Int8);
        registry.add_cast_extension(Arc::new(Bool8Casts))?;
        let extension = registry.cast_extension().unwrap();
        let options = options(extension.as_ref());

        let error = extension
            .can_cast_fields(&bool8, &Field::new("value", DataType::Utf8, true), &options)
            .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("Multiple registered cast extensions")
        );
        assert!(error.to_string().contains(Bool8::NAME));
        Ok(())
    }
}
