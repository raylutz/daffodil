# schemaclass

`schemaclass` is a decorator for defining **tabular schemas** using Python class syntax.
A schemaclass defines column names, intended column types, default values, and optional
initializer metadata in a single, inspectable structure.

The decorated class is used only as a schema descriptor and is not intended to be instantiated.

## Defining a schema

    from daffodil.lib.schemaclass import schemaclass

    @schemaclass
    class MarksSchema:
        ballot_id: str = ''
        contest: str = ''
        vote: int = 0

Each annotated attribute defines a column:

- the attribute name is the column name
- the annotation defines the intended type of non-empty values
- the attribute value defines the default value

All columns must have both a type annotation and a default value.

## Schema requirements

A valid schemaclass must:

- define `__annotations__` as a non-empty dictionary
- provide a default value for every annotated field

Invalid schemas raise an exception at decoration time.

## Prevented instantiation

A schemaclass cannot be instantiated.

Attempting to construct an instance raises a `TypeError`.
This prevents accidental use of schemas as data objects.

## default_record()

Generate a new record dictionary initialized from the schema’s default values.

    record = MySchema.default_record()

A new dictionary is returned on each call.

For scalar and immutable defaults (such as `str`, `int`, `float`, or `tuple`),
the value is copied by reference.

For mutable defaults (such as `list`, `dict`, or `set`), a shallow copy is
created so that each record receives an independent object.

No validation or type conversion is performed.

## dtypes dictionary generation

A schemaclass provides a class method to generate a `dtypes` dictionary:

    dtypes = MarksSchema.dtypes_dict()

Result:

    {
        'ballot_id': str,
        'contest': str,
        'vote': int
    }

The `dtypes` dictionary maps column names to intended Python types.
These types describe non-empty values only.

## Type semantics

Type annotations describe the intended type of non-empty values.
They are not enforced at runtime.

Defaults may use empty values such as `''`, `()`, `{}`, or `[]`
to represent missing or not-present data.

This reflects common CSV and spreadsheet conventions.

## Initializer metadata

A schemaclass may define additional class attributes used as initializer
metadata by consuming systems.

Initializer metadata does not define columns and is excluded from
default records and `dtypes` dictionaries.

### `__keyfield__`

The `__keyfield__` attribute identifies one or more columns that should
be treated as a key.

Example:

    @schemaclass
    class MarksSchema:
        __keyfield__ = 'ballot_id'

        ballot_id: str = ''
        contest: str = ''
        vote: int = 0

Composite keys may be specified as a tuple or list of column names:

    __keyfield__ = ('ballot_id', 'contest')

The interpretation of `__keyfield__` is handled by the consuming system.

## Marker attribute

Decorated classes include the attribute:

    __is_schemaclass__ = True

This allows consuming code to identify schema classes explicitly.

## File organization

There are three options normally used for defining schemas:

### One schema per file in models/

Schemas for each table can be derined in separate files, with either one schema definition per file or several related schemas.

    models/
        marks_schema.py
        contests_schema.py
        ballots_schema.py
        
### One file in models/

All schemas can be included in a single file to reduce file proliferation. For example:

    models/schema.py
    
    
### Definition near usage

It may make sense to define the schema near where it is used, particularly when the schema is not shared widely.    
    
    

## Summary

`schemaclass` provides a minimal, explicit mechanism for defining
tabular schemas with:

- column names
- intended types
- default values
- initializer metadata

It separates schema description from data storage and processing,
allowing consuming systems such as Daffodil to handle typing,
conversion, indexing, and validation independently.
