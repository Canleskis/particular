/// Derive macro generating an implementation of the trait `Particle`.
#[proc_macro_derive(Particle)]
pub fn particle_derive(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let ast = syn::parse(input);

    impl_particle(ast).unwrap_or_else(|e| syn::Error::to_compile_error(&e).into())
}

fn impl_particle(input: syn::Result<syn::DeriveInput>) -> syn::Result<proc_macro::TokenStream> {
    let input = input?;

    let name = input.ident;
    let (pty, sty) = get_field_types(input.data)?;
    let (impl_generics, ty_generics, where_clause) = input.generics.split_for_impl();

    Ok(quote::quote! {
        impl #impl_generics Particle for #name #ty_generics #where_clause {
            type Scalar = #sty;

            type Vector = #pty;

            #[inline]
            fn position(&self) -> #pty {
                self.position
            }

            #[inline]
            fn mu(&self) -> #sty {
                self.mu
            }
        }
    }
    .into())
}

fn get_field_types(data: syn::Data) -> syn::Result<(syn::Type, syn::Type)> {
    match &data {
        syn::Data::Struct(struct_data) => Ok((
            get_type_of(struct_data, "position")?,
            get_type_of(struct_data, "mu")?,
        )),
        syn::Data::Enum(enum_data) => Err(syn::Error::new_spanned(
            enum_data.enum_token,
            "the `Particle` trait can only be derived for struct types",
        )),
        syn::Data::Union(union_data) => Err(syn::Error::new_spanned(
            union_data.union_token,
            "the `Particle` trait can only be derived for struct types",
        )),
    }
}

fn get_type_of(struct_data: &syn::DataStruct, field_name: &str) -> syn::Result<syn::Type> {
    struct_data
        .fields
        .iter()
        .find_map(|field| (field.ident.as_ref()? == field_name).then(|| field.ty.clone()))
        .ok_or_else(|| {
            syn::Error::new_spanned(&struct_data.fields, format!("no {field_name} field"))
        })
}
