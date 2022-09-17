extern crate proc_macro;

use proc_macro::TokenStream;
use quote::quote;
use syn::{spanned::Spanned, Data};

/// Derive macro generating an implementation of the trait `Particle`.
#[proc_macro_derive(Particle)]
pub fn particle_derive(input: TokenStream) -> TokenStream {
    let ast = syn::parse(input).unwrap();

    impl_particle(ast).unwrap_or_else(|e| syn::Error::to_compile_error(&e).into())
}

fn get_position_type(data: syn::Data) -> syn::Result<syn::Type> {
    match data {
        Data::Struct(struct_data) => {
            let fields_span = struct_data.fields.span();

            struct_data
                .fields
                .into_iter()
                .find(|field| match field.ident.as_ref() {
                    Some(ident) => *ident == "position",
                    None => false,
                })
                .map_or_else(
                    || Err(syn::Error::new(fields_span, "No position field")),
                    |field| Ok(field.ty),
                )
        }
        Data::Enum(enum_data) => Err(syn::Error::new_spanned(
            enum_data.enum_token,
            "An enum cannot represent a Particle",
        )),
        Data::Union(union_data) => Err(syn::Error::new_spanned(
            union_data.union_token,
            "A union cannot represent a Particle",
        )),
    }
}

fn impl_particle(ast: syn::DeriveInput) -> syn::Result<TokenStream> {
    let name = ast.ident;
    let ty = get_position_type(ast.data)?;
    let (impl_generics, ty_generics, where_clause) = ast.generics.split_for_impl();

    Ok(quote! {
        impl #impl_generics Particle for #name #ty_generics #where_clause {
            type Vector = #ty;

            fn position(&self) -> Self::Vector {
                self.position
            }

            fn mu(&self) -> f32 {
                self.mu
            }
        }
    }
    .into())
}
