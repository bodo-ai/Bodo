# GET

- `#!sql GET(arr, idx)`
- `#!sql GET(object, field)`
- `#!sql arr[idx]`
- `#!sql object[field]`

Returns the element found at the specified index in the array, or the specified field of an object.

When indexing into an array: indexing is 0 based, not 1 based. Returns `#!sql NULL` if the index is outside of the boundaries of the array. The index must be an integer.

When retrieving a field from an object: the field name must be a string. If the object is a struct, the field name must be a string literal. If the object is a map, it can be a non-constant string. Returns `#!sql NULL` if the field name is not found. Field name
matching is case-sensitive.


