/// Derive macro generating an implementation of the trait `Particle`.
#[proc_macro_derive(Particle, attributes(dim))]
pub fn particle_derive(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    impl_particle(syn::parse(input)).unwrap_or_else(|e| syn::Error::to_compile_error(&e).into())
}

fn impl_particle(input: syn::Result<syn::DeriveInput>) -> syn::Result<proc_macro::TokenStream> {
    let input = input?;

    let (impl_generics, ty_generics, where_clause) = input.generics.split_for_impl();
    let name = &input.ident;
    let scalar_ty = match &input.data {
        syn::Data::Struct(data_struct) => Ok(get_field("mu", data_struct)
            .ok_or_else(|| syn::Error::new_spanned(&data_struct.fields, "no `mu` field"))?
            .ty),
        _ => Err(syn::Error::new_spanned(
            &input.generics,
            "the `Particle` trait can only be derived for struct types",
        )),
    }?;
    let dim = get_attribute("dim", &input.attrs)
        .ok_or_else(|| {
            syn::Error::new_spanned(
                &input.generics,
                "no `#[dim]` attribute\n\
                add `#[dim(arg)]` with the dimension of the particle as the argument",
            )
        })?
        .parse_args::<syn::LitInt>()?
        .base10_parse::<usize>()?;

    Ok(quote::quote! {
        impl #impl_generics Particle for #name #ty_generics #where_clause {
            type Array = [#scalar_ty; #dim];

            #[inline]
            fn position(&self) -> Self::Array {
                self.position.into()
            }

            #[inline]
            fn mu(&self) -> #scalar_ty {
                self.mu
            }
        }
    }
    .into())
}

fn get_field(name: &str, struct_data: &syn::DataStruct) -> Option<syn::Field> {
    struct_data
        .fields
        .iter()
        .find_map(|field| (field.ident.as_ref()? == name).then_some(field))
        .cloned()
}

fn get_attribute(name: &str, attrs: &[syn::Attribute]) -> Option<syn::Attribute> {
    attrs
        .iter()
        .find(|attr| attr.path().segments[0].ident == name)
        .cloned()
}
