pub fn impl_position(input: syn::Result<syn::DeriveInput>) -> syn::Result<proc_macro::TokenStream> {
    let mut input = input?;

    let position_ty = match &input.data {
        syn::Data::Struct(data_struct) => Ok(&crate::get_field("position", data_struct)
            .ok_or_else(|| syn::Error::new_spanned(&data_struct.fields, "no `position` field"))?
            .ty),
        _ => Err(syn::Error::new_spanned(
            &input.generics,
            "the `Position` trait can only be derived for struct types",
        )),
    }?;

    input
        .generics
        .where_clause
        .get_or_insert_with(|| syn::WhereClause {
            where_token: Default::default(),
            predicates: Default::default(),
        })
        .predicates
        .push(syn::parse_quote! {
            #position_ty: ::core::clone::Clone
        });

    let (impl_generics, ty_generics, where_clause) = input.generics.split_for_impl();
    let name = &input.ident;

    Ok(quote::quote! {
        impl #impl_generics Position for #name #ty_generics #where_clause {
            type Vector = #position_ty;

            #[inline]
            fn position(&self) -> Self::Vector {
                self.position.clone()
            }
        }
    }
    .into())
}

pub fn impl_mass(input: syn::Result<syn::DeriveInput>) -> syn::Result<proc_macro::TokenStream> {
    let mut input = input?;

    let field = match &input.data {
        syn::Data::Struct(data_struct) => Ok(crate::get_field("mu", data_struct)
            .or_else(|| crate::get_field("mass", data_struct))
            .ok_or_else(|| {
                syn::Error::new_spanned(&data_struct.fields, "no `mu` or `mass` field")
            })?),
        _ => Err(syn::Error::new(
            proc_macro2::Span::call_site(),
            "the `ToPointMass` trait can only be derived for struct types",
        )),
    }?;

    let field_type = &field.ty;
    let field_ident = field.ident.as_ref().unwrap();
    let constant = crate::get_attribute("G", &field.attrs)
        .map(|attr| attr.meta.require_name_value().map(|meta| &meta.value))
        .transpose()?;

    let mass_method = match (field_ident == "mu", constant.is_some()) {
        (true, true) => quote::quote! {
            #[inline]
            fn mass(&self) -> Self::Scalar {
                self.mu.clone() / #constant
            }
        },
        _ => identity_method(quote::format_ident!("mass"), field_ident),
    };

    let mu_method = match (field_ident == "mass", constant.is_some()) {
        (true, true) => quote::quote! {
            #[inline]
            fn mu(&self) -> Self::Scalar {
                self.mass.clone() * #constant
            }
        },
        _ => identity_method(quote::format_ident!("mu"), field_ident),
    };

    input
        .generics
        .where_clause
        .get_or_insert_with(|| syn::WhereClause {
            where_token: Default::default(),
            predicates: Default::default(),
        })
        .predicates
        .push(syn::parse_quote! {
            #field_type: ::core::clone::Clone
        });

    let (impl_generics, ty_generics, where_clause) = input.generics.split_for_impl();
    let name = &input.ident;

    Ok(quote::quote! {
        impl #impl_generics Mass for #name #ty_generics #where_clause {
            type Scalar = #field_type;

            #mass_method

            #mu_method
        }
    }
    .into())
}

fn identity_method(method: syn::Ident, ident: &syn::Ident) -> proc_macro2::TokenStream {
    quote::quote! {
        #[inline]
        fn #method(&self) -> Self::Scalar {
            self.#ident.clone()
        }
    }
}
