mod gravity;

/// Derive macro generating an implementation of the `Position` trait.
#[proc_macro_derive(Position)]
pub fn position_derive(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    gravity::impl_position(syn::parse(input))
        .unwrap_or_else(|e| syn::Error::to_compile_error(&e).into())
}

/// Derive macro generating an implementation of the `Mass` trait.
#[proc_macro_derive(Mass, attributes(G))]
pub fn mass_derive(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    gravity::impl_mass(syn::parse(input))
        .unwrap_or_else(|e| syn::Error::to_compile_error(&e).into())
}

fn get_field<'a>(name: &str, struct_data: &'a syn::DataStruct) -> Option<&'a syn::Field> {
    struct_data
        .fields
        .iter()
        .find_map(|field| (field.ident.as_ref()? == name).then_some(field))
}

fn get_attribute<'a>(name: &str, attrs: &'a [syn::Attribute]) -> Option<&'a syn::Attribute> {
    attrs.iter().find(|attr| attr.path().is_ident(name))
}
