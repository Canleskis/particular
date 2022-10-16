use proc_macro::TokenStream;
use quote::{quote, ToTokens};
use syn::{parse::Parser, punctuated, spanned::Spanned, Data};

/// Attribute macro generating a derivation of the trait `Particle`.
#[proc_macro_attribute]
pub fn particle(attr: TokenStream, input: TokenStream) -> TokenStream {
    let item = syn::parse_macro_input!(input as syn::ItemStruct);

    derive_particle(attr, item).unwrap_or_else(|e| syn::Error::to_compile_error(&e).into())
}

fn derive_particle(attr: TokenStream, mut item: syn::ItemStruct) -> syn::Result<TokenStream> {
    let dim = parse_attr(attr, item.semi_token.span())?;

    item.attrs.extend([
        syn::parse_quote! {
            #[derive(Particle)]
        },
        syn::parse_quote! {
            #[dim(#dim)]
        },
    ]);

    Ok(item.into_token_stream().into())
}

fn parse_attr(attr: TokenStream, span: syn::__private::Span) -> syn::Result<usize> {
    punctuated::Punctuated::<syn::LitInt, syn::Token![,]>::parse_terminated
        .parse(attr)?
        .first()
        .ok_or_else(|| {
            syn::Error::new(
                span,
                "no dimension provided for the particle\n\
                add the dimension to the attribute",
            )
        })?
        .base10_parse::<usize>()
}

/// Derive macro generating an implementation of the trait `Particle`.
#[proc_macro_derive(Particle, attributes(dim))]
pub fn particle_derive(input: TokenStream) -> TokenStream {
    let ast = syn::parse(input);

    impl_particle(ast).unwrap_or_else(|e| syn::Error::to_compile_error(&e).into())
}

fn impl_particle(input: syn::Result<syn::DeriveInput>) -> syn::Result<TokenStream> {
    let input = input?;

    let (dim, ty) = (
        get_dimension(input.attrs, input.ident.span())?,
        get_position_type(input.data)?,
    );

    let name = input.ident;
    let (impl_generics, ty_generics, where_clause) = input.generics.split_for_impl();

    Ok(quote! {
        impl #impl_generics Particle for #name #ty_generics #where_clause {
            type Vector = VectorDescriptor<#dim, #ty>;

            #[inline]
            fn position(&self) -> #ty {
                self.position
            }

            #[inline]
            fn mu(&self) -> f32 {
                self.mu
            }
        }
    }
    .into())
}

fn get_position_type(data: syn::Data) -> syn::Result<syn::Type> {
    match data {
        Data::Struct(struct_data) => {
            let fields_span = struct_data.fields.span();

            struct_data
                .fields
                .into_iter()
                .find_map(|field| match field.ident {
                    Some(ident) => (ident == "position").then_some(field.ty),
                    None => None,
                })
                .ok_or_else(|| syn::Error::new(fields_span, "no position field"))
        }
        Data::Enum(enum_data) => Err(syn::Error::new_spanned(
            enum_data.enum_token,
            "an enum cannot represent a Particle",
        )),
        Data::Union(union_data) => Err(syn::Error::new_spanned(
            union_data.union_token,
            "a union cannot represent a Particle",
        )),
    }
}

fn get_dimension(attrs: Vec<syn::Attribute>, span: syn::__private::Span) -> syn::Result<usize> {
    attrs
        .iter()
        .find(|attr| attr.path.segments.len() == 1 && attr.path.segments[0].ident == "dim")
        .map_or_else(
            || {
                Err(syn::Error::new(
                    span,
                    "no #[dim] attribute\n\
                    add #[dim(arg)] with \
                    the dimension of the \
                    particle as the argument",
                ))
            },
            |attr| attr.parse_args::<syn::LitInt>()?.base10_parse::<usize>(),
        )
}
