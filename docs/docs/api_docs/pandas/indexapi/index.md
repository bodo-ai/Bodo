# Index objects

## Index

### Properties

- \[`pd.Index.name`\][pdindexname]
- \[`pd.Index.names`\][pdindexnames]
- \[`pd.Index.shape`\][pdindexshape]
- \[`pd.Index.size`\][pdindexsize]
- \[`pd.Index.empty`\][pdindexempty]
- \[`pd.Index.is_monotonic_increasing`\][pdindexis_monotonic_increasing]
- \[`pd.Index.is_monotonic_decreasing`\][pdindexis_monotonic_decreasing]
- \[`pd.Index.values`\][pdindexvalues]
- \[`pd.Index.nbytes`\][pdindexnbytes]
- \[`pd.Index.ndim`\][pdindexndim]
- \[`pd.Index.nlevels`\][pdindexnlevels]
- \[`pd.Index.dtype`\][pdindexdtype]
- \[`pd.Index.inferred_type`\][pdindexinferred_type]
- \[`pd.Index.is_all_dates`\][pdindexis_all_dates]
- \[`pd.Index.T`\][pdindext]

### Type information

- \[`pd.Index.is_numeric`\][pdindexis_numeric]
- \[`pd.Index.is_integer`\][pdindexis_integer]
- \[`pd.Index.is_floating`\][pdindexis_floating]
- \[`pd.Index.is_boolean`\][pdindexis_boolean]
- \[`pd.Index.is_categorical`\][pdindexis_categorical]
- \[`pd.Index.is_interval`\][pdindexis_interval]
- \[`pd.Index.is_object`\][pdindexis_object]

### Modifications and computations

- \[`pd.Index.copy`\][pdindexcopy]
- \[`pd.Index.get_loc`\][pdindexget_loc]
- \[`pd.Index.take`\][pdindextake]
- \[`pd.Index.min`\][pdindexmin]
- \[`pd.Index.max`\][pdindexmax]
- \[`pd.Index.rename`\][pdindexrename]
- \[`pd.Index.duplicated`\][pdindexduplicated]
- \[`pd.Index.drop_duplicates`\][pdindexdrop_duplicates]
- \[`pd.Index.isin`\][pdindexisin]
- \[`pd.Index.unique`\][pdindexunique]
- \[`pd.Index.nunique`\][pdindexnunique]
- \[`pd.Index.sort_values`\][pdindexsort_values]
- \[`pd.Index.argsort`\][pdindexargsort]
- \[`pd.Index.all`\][pdindexall]
- \[`pd.Index.any`\][pdindexany]
- \[`pd.Index.argmax`\][pdindexargmax]
- \[`pd.Index.argmin`\][pdindexargmin]
- \[`pd.Index.where`\][pdindexwhere]
- \[`pd.Index.putmask`\][pdindexputmask]
- \[`pd.Index.union`\][pdindexunion]
- \[`pd.Index.intersection`\][pdindexintersection]
- \[`pd.Index.difference`\][pdindexdifference]
- \[`pd.Index.symmetric_difference`\][pdindexsymmetric_difference]
- \[`pd.Index.repeat`\][pdindexrepeat]

### Missing values

- \[`pd.Index.isna`\][pdindexisna]
- \[`pd.Index.isnull`\][pdindexisnull]

### Conversion

- \[`pd.Index.map`\][pdindexmap]
- \[`pd.Index.to_series`\][pdindexto_series]
- \[`pd.Index.to_frame`\][pdindexto_frame]
- \[`pd.Index.to_numpy`\][pdindexto_numpy]
- \[`pd.Index.to_list`\][pdindexto_list]
- \[`pd.Index.tolist`\][pdindextolist]

## Numeric Index

Numeric index objects `RangeIndex`, `Int64Index`, `UInt64Index` and
`Float64Index` are supported as index to dataframes and series.
Constructing them in Bodo functions, passing them to Bodo functions (unboxing),
and returning them from Bodo functions (boxing) are also supported.

- \[`pd.RangeIndex`\][pdrangeindex]
- \[`pd.Int64Index`\][pdint64index]
- \[`pd.UInt64Index`\][pduint64index]
- \[`pd.Float64Index`\][pdfloat64index]

## DatetimeIndex

`DatetimeIndex` objects are supported. They can be constructed,
boxed/unboxed, and set as index to dataframes and series.

- \[`pd.DateTimeIndex`\][pddatetimeindex]
- \[`pd.DateTimeIndex.year`\][pddatetimeindexyear]
- \[`pd.DateTimeIndex.month`\][pddatetimeindexmonth]
- \[`pd.DateTimeIndex.day`\][pddatetimeindexday]
- \[`pd.DateTimeIndex.hour`\][pddatetimeindexhour]
- \[`pd.DateTimeIndex.minute`\][pddatetimeindexminute]
- \[`pd.DateTimeIndex.second`\][pddatetimeindexsecond]
- \[`pd.DateTimeIndex.microsecond`\][pddatetimeindexmicrosecond]
- \[`pd.DateTimeIndex.nanosecond`\][pddatetimeindexnanosecond]
- \[`pd.DateTimeIndex.date`\][pddatetimeindexdate]
- \[`pd.DateTimeIndex.dayofyear`\][pddatetimeindexdayofyear]
- \[`pd.DateTimeIndex.day_of_year`\][pddatetimeindexday_of_year]
- \[`pd.DateTimeIndex.dayofweek`\][pddatetimeindexdayofweek]
- \[`pd.DateTimeIndex.day_of_week`\][pddatetimeindexday_of_week]
- \[`pd.DateTimeIndex.is_leap_year`\][pddatetimeindexis_leap_year]
- \[`pd.DateTimeIndex.is_month_start`\][pddatetimeindexis_month_start]
- \[`pd.DateTimeIndex.is_month_end`\][pddatetimeindexis_month_end]
- \[`pd.DateTimeIndex.is_quarter_start`\][pddatetimeindexis_quarter_start]
- \[`pd.DateTimeIndex.is_quarter_end`\][pddatetimeindexis_quarter_end]
- \[`pd.DateTimeIndex.is_year_start`\][pddatetimeindexis_year_start]
- \[`pd.DateTimeIndex.is_year_end`\][pddatetimeindexis_year_end]
- \[`pd.DateTimeIndex.week`\][pddatetimeindexweek]
- \[`pd.DateTimeIndex.weekday`\][pddatetimeindexweekday]
- \[`pd.DateTimeIndex.weekofyear`\][pddatetimeindexweekofyear]
- \[`pd.DateTimeIndex.quarter`\][pddatetimeindexquarter]

Subtraction of `Timestamp` from `DatetimeIndex` and vice versa
is supported.

Comparison operators `==`, `!=`, `>=`, `>`, `<=`, `<` between
`DatetimeIndex` and a string of datetime
are supported.

## TimedeltaIndex

`TimedeltaIndex` objects are supported. They can be constructed,
boxed/unboxed, and set as index to dataframes and series.

- \[`pd.TimedeltaIndex`\][pdtimedeltaindex]
- \[`pd.TimedeltaIndex.days`\][pdtimedeltaindexdays]
- \[`pd.TimedeltaIndex.seconds`\][pdtimedeltaindexseconds]
- \[`pd.TimedeltaIndex.microseconds`\][pdtimedeltaindexmicroseconds]
- \[`pd.TimedeltaIndex.nanoseconds`\][pdtimedeltaindexnanoseconds]

## PeriodIndex

`PeriodIndex` objects can be
boxed/unboxed and set as index to dataframes and series.
Operations on them will be supported in upcoming releases.

## BinaryIndex

`BinaryIndex` objects can be
boxed/unboxed and set as index to dataframes and series.
Operations on them will be supported in upcoming releases.

## MultiIndex

- \[`pd.MultiIndex.from_product`\][pdmultiindexfrom_product]
