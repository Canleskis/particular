use quote::quote;
use proc_macro::TokenStream;
use syn::{spanned::Spanned, Data};

/// Derive macro generating an implementation of the trait `Particle`.
#[proc_macro_derive(Particle, attributes(dim))]
pub fn particle_derive(input: TokenStream) -> TokenStream {
    let ast = syn::parse(input).unwrap();

    impl_particle(ast).unwrap_or_else(|e| syn::Error::to_compile_error(&e).into())
}

fn impl_particle(input: syn::DeriveInput) -> syn::Result<TokenStream> {
    let (dim, ty, pos) = get_position(input.data)?;
    
    let name = input.ident;
    let (impl_generics, ty_generics, where_clause) = input.generics.split_for_impl();

    Ok(quote! {
        impl #impl_generics Particle for #name #ty_generics #where_clause {
            type Vector = VectorDescriptor<#dim, #ty>;

            #[inline]
            fn position(&self) -> #ty {
                self.#pos
            }

            #[inline]
            fn mu(&self) -> f32 {
                self.mu
            }
        }
    }
    .into())
}

fn get_position(data: syn::Data) -> syn::Result<(usize, syn::Type, syn::Ident)> {
    match data {
        Data::Struct(struct_data) => {
            struct_data
                .fields
                .iter()
                .find_map(|field| {
                    field.attrs.iter().find_map(|attr| {
                        (attr.path.segments.len() == 1 && attr.path.segments[0].ident == "dim")
                            .then_some((attr, field.clone()))
                    })
                })
                .map_or(Err(syn::Error::new(
                    struct_data.fields.span(),
                    "No field tagged with the #[dim] attribute\n\
                    add #[dim(arg)] to the position field with \
                    the dimension of its vector as the argument",
                )), |(attr, field)| {
                    Ok((
                        attr.parse_args::<syn::LitInt>()?.base10_parse::<usize>()?,
                        field.ty,
                        field.ident.unwrap(),
                    ))
                })
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