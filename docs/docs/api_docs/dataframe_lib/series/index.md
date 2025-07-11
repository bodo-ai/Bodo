# Series API
The Bodo DataFrame Library supports Pandas Series methods and accessors that are listed below. They can be accessed through `BodoSeries` and follow the same behavior as their Pandas equivalents. For details on usage, we link to the corresponding Pandas documentation.

!!! note
    If the user code encounters an unsupported Pandas API or an unsupported parameter, Bodo
	 DataFrame Library gracefully falls back to native Pandas. See [overview][overview] of 
	 the Bodo DataFrame Library for more info.

## Computations / descriptive stats
- [`bodo.pandas.BodoSeries.abs`][bodoseriesabs]
- [`bodo.pandas.BodoSeries.clip`][bodoseriesclip]
- [`bodo.pandas.BodoSeries.round`][bodoseriesround]
!!! note
    For the following reduction methods, only default parameters are currently supported.
- [`bodo.pandas.BodoSeries.count`][bodoseriescount]
- [`bodo.pandas.BodoSeries.max`][bodoseriesmax]
- [`bodo.pandas.BodoSeries.min`][bodoseriesmin]
- [`bodo.pandas.BodoSeries.product`][bodoseriesproduct]
- [`bodo.pandas.BodoSeries.sum`][bodoseriessum]

---

## Datetimelike properties

!!! note
    Input must be a Series of `datetime-like` data.


### Datetime properties
!!! note
    For missing datetime values (`NaT`), Bodo's datetime predicate accessors (e.g., `.is_month_end`, `.is_leap_year`) return `<NA>` to preserve nullability, whereas Pandas returns `False`.
- [`bodo.pandas.BodoSeries.dt.year`][bodoseriesdtyear]
- [`bodo.pandas.BodoSeries.dt.month`][bodoseriesdtmonth]
- [`bodo.pandas.BodoSeries.dt.day`][bodoseriesdtday]
- [`bodo.pandas.BodoSeries.dt.days`][bodoseriesdtdays]
- [`bodo.pandas.BodoSeries.dt.hour`][bodoseriesdthour]
- [`bodo.pandas.BodoSeries.dt.minute`][bodoseriesdtminute]
- [`bodo.pandas.BodoSeries.dt.second`][bodoseriesdtsecond]
- [`bodo.pandas.BodoSeries.dt.seconds`][bodoseriesdtseconds]
- [`bodo.pandas.BodoSeries.dt.microsecond`][bodoseriesdtmicrosecond]
- [`bodo.pandas.BodoSeries.dt.microseconds`][bodoseriesdtmicroseconds]
- [`bodo.pandas.BodoSeries.dt.nanosecond`][bodoseriesdtnanosecond]
- [`bodo.pandas.BodoSeries.dt.nanoseconds`][bodoseriesdtnanoseconds]
- [`bodo.pandas.BodoSeries.dt.dayofweek`][bodoseriesdtdayofweek]
- [`bodo.pandas.BodoSeries.dt.day_of_week`][bodoseriesdtday_of_week]
- [`bodo.pandas.BodoSeries.dt.weekday`][bodoseriesdtweekday]
- [`bodo.pandas.BodoSeries.dt.dayofyear`][bodoseriesdtdayofyear]
- [`bodo.pandas.BodoSeries.dt.day_of_year`][bodoseriesdtday_of_year]
- [`bodo.pandas.BodoSeries.dt.daysinmonth`][bodoseriesdtdaysinmonth]
- [`bodo.pandas.BodoSeries.dt.days_in_month`][bodoseriesdtdays_in_month]
- [`bodo.pandas.BodoSeries.dt.date`][bodoseriesdtdate]
- [`bodo.pandas.BodoSeries.dt.time`][bodoseriesdttime]
- [`bodo.pandas.BodoSeries.dt.quarter`][bodoseriesdtquarter]
- [`bodo.pandas.BodoSeries.dt.is_month_start`][bodoseriesdtis_month_start]
- [`bodo.pandas.BodoSeries.dt.is_month_end`][bodoseriesdtis_month_end]
- [`bodo.pandas.BodoSeries.dt.is_quarter_start`][bodoseriesdtis_quarter_start]
- [`bodo.pandas.BodoSeries.dt.is_quarter_end`][bodoseriesdtis_quarter_end]
- [`bodo.pandas.BodoSeries.dt.is_year_start`][bodoseriesdtis_year_start]
- [`bodo.pandas.BodoSeries.dt.is_year_end`][bodoseriesdtis_year_end]
- [`bodo.pandas.BodoSeries.dt.is_leap_year`][bodoseriesdtis_leap_year]
- [`bodo.pandas.BodoSeries.dt.components`][bodoseriesdtcomponents]

---

### Datetime methods
!!! note
    Locale format must be strict: The locale parameter in `month_name` and `day_name` must 
	follow the exact system locale naming convention (e.g., "pt_BR.UTF-8" or "en_US.utf-8"). 
	Variants like "pt_BR.utf8" may not be recognized and trigger an error.

- [`bodo.pandas.BodoSeries.dt.normalize`][bodoseriesdtnormalize]
- [`bodo.pandas.BodoSeries.dt.floor`][bodoseriesdtfloor]
- [`bodo.pandas.BodoSeries.dt.ceil`][bodoseriesdtceil]
- [`bodo.pandas.BodoSeries.dt.month_name`][bodoseriesdtmonth_name]
- [`bodo.pandas.BodoSeries.dt.day_name`][bodoseriesdtday_name]
- [`bodo.pandas.BodoSeries.dt.round`][bodoseriesdtround]
- [`bodo.pandas.BodoSeries.dt.total_seconds`][bodoseriesdttotal_seconds]
- [`bodo.pandas.BodoSeries.dt.isocalendar`][bodoseriesdtisocalendar]

!!! note
    Bodo currently only supports "NaT" for the ambiguous parameter in `tz_localize`. 
	"raise", "infer", or boolean arrays are not supported and will trigger a fallback to 
	Pandas. Similarly, for the nonexistent parameter, 
    "raise" is not supported and will trigger a fallback. Due to these limitations, the
	default behavior in Bodo is `ambiguous="NaT"` and `nonexistent="NaT"`.


- [`bodo.pandas.BodoSeries.dt.tz_localize`][bodoseriesdttz_localize]


---

## Function application
- [`bodo.pandas.BodoSeries.agg`][bodoseriesagg]
- [`bodo.pandas.BodoSeries.aggregate`][bodoseriesaggregate]
- [`bodo.pandas.BodoSeries.map`][bodoseriesmap]

---

## Missing data handling
- [`bodo.pandas.BodoSeries.isna`][bodoseriesisna]
- [`bodo.pandas.BodoSeries.isnull`][bodoseriesisnull]
- [`bodo.pandas.BodoSeries.notna`][bodoseriesnotna]
- [`bodo.pandas.BodoSeries.notnull`][bodoseriesnotnull]
- [`bodo.pandas.BodoSeries.replace`][bodoseriesreplace]

---

## Reindexing / Selection / Label manipulation
- [`bodo.pandas.BodoSeries.head`][bodoserieshead]
- [`bodo.pandas.BodoSeries.isin`][bodoseriesisin]

---

## Reshaping, sorting
- [`bodo.pandas.BodoSeries.sort_values`][bodoseriessortvalues]

---

## String handling

- [`bodo.pandas.BodoSeries.str.capitalize`][bodoseriesstrcapitalize]
- [`bodo.pandas.BodoSeries.str.casefold`][bodoseriesstrcasefold]
!!! note
    `cat` falls back to Pandas when the others parameter is not specified (i.e., `others=None`).
- [`bodo.pandas.BodoSeries.str.cat`][bodoseriesstrcat]
- [`bodo.pandas.BodoSeries.str.center`][bodoseriesstrcenter]
- [`bodo.pandas.BodoSeries.str.contains`][bodoseriesstrcontains]
- [`bodo.pandas.BodoSeries.str.count`][bodoseriesstrcount]
- [`bodo.pandas.BodoSeries.str.decode`][bodoseriesstrdecode]
- [`bodo.pandas.BodoSeries.str.encode`][bodoseriesstrencode]
- [`bodo.pandas.BodoSeries.str.endswith`][bodoseriesstrendswith]
- [`bodo.pandas.BodoSeries.str.extract`][bodoseriesstrextract]
- [`bodo.pandas.BodoSeries.str.find`][bodoseriesstrfind]
- [`bodo.pandas.BodoSeries.str.findall`][bodoseriesstrfindall]
- [`bodo.pandas.BodoSeries.str.fullmatch`][bodoseriesstrfullmatch]
- [`bodo.pandas.BodoSeries.str.get`][bodoseriesstrget]
- [`bodo.pandas.BodoSeries.str.index`][bodoseriesstrindex]
- [`bodo.pandas.BodoSeries.str.isalnum`][bodoseriesstrisalnum]
- [`bodo.pandas.BodoSeries.str.isalpha`][bodoseriesstrisalpha]
- [`bodo.pandas.BodoSeries.str.isdecimal`][bodoseriesstrisdecimal]
- [`bodo.pandas.BodoSeries.str.isdigit`][bodoseriesstrisdigit]
- [`bodo.pandas.BodoSeries.str.islower`][bodoseriesstrislower]
- [`bodo.pandas.BodoSeries.str.isnumeric`][bodoseriesstrisnumeric]
- [`bodo.pandas.BodoSeries.str.isspace`][bodoseriesstrisspace]
- [`bodo.pandas.BodoSeries.str.istitle`][bodoseriesstristitle]
- [`bodo.pandas.BodoSeries.str.isupper`][bodoseriesstrisupper]
- [`bodo.pandas.BodoSeries.str.join`][bodoseriesstrjoin]
- [`bodo.pandas.BodoSeries.str.len`][bodoseriesstrlen]
- [`bodo.pandas.BodoSeries.str.ljust`][bodoseriesstrljust]
- [`bodo.pandas.BodoSeries.str.lower`][bodoseriesstrlower]
- [`bodo.pandas.BodoSeries.str.lstrip`][bodoseriesstrlstrip]
- [`bodo.pandas.BodoSeries.str.match`][bodoseriesstrmatch]
- [`bodo.pandas.BodoSeries.str.normalize`][bodoseriesstrnormalize]
- [`bodo.pandas.BodoSeries.str.pad`][bodoseriesstrpad]
- [`bodo.pandas.BodoSeries.str.partition`][bodoseriesstrpartition]
- [`bodo.pandas.BodoSeries.str.removeprefix`][bodoseriesstrremoveprefix]
- [`bodo.pandas.BodoSeries.str.removesuffix`][bodoseriesstrremovesuffix]
- [`bodo.pandas.BodoSeries.str.repeat`][bodoseriesstrrepeat]
- [`bodo.pandas.BodoSeries.str.replace`][bodoseriesstrreplace]
- [`bodo.pandas.BodoSeries.str.rfind`][bodoseriesstrrfind]
- [`bodo.pandas.BodoSeries.str.rindex`][bodoseriesstrrindex]
- [`bodo.pandas.BodoSeries.str.rjust`][bodoseriesstrrjust]
- [`bodo.pandas.BodoSeries.str.rpartition`][bodoseriesstrrpartition]
- [`bodo.pandas.BodoSeries.str.rsplit`][bodoseriesstrrsplit]
- [`bodo.pandas.BodoSeries.str.rstrip`][bodoseriesstrrstrip]
- [`bodo.pandas.BodoSeries.str.slice`][bodoseriesstrslice]
- [`bodo.pandas.BodoSeries.str.slice_replace`][bodoseriesstrslicereplace]
- [`bodo.pandas.BodoSeries.str.split`][bodoseriesstrsplit]
- [`bodo.pandas.BodoSeries.str.startswith`][bodoseriesstrstartswith]
- [`bodo.pandas.BodoSeries.str.strip`][bodoseriesstrstrip]
- [`bodo.pandas.BodoSeries.str.swapcase`][bodoseriesstrswapcase]
- [`bodo.pandas.BodoSeries.str.title`][bodoseriesstrtitle]
- [`bodo.pandas.BodoSeries.str.translate`][bodoseriesstrtranslate]
- [`bodo.pandas.BodoSeries.str.upper`][bodoseriesstrupper]
- [`bodo.pandas.BodoSeries.str.wrap`][bodoseriesstrwrap]
- [`bodo.pandas.BodoSeries.str.zfill`][bodoseriesstrzfill]


[bodoserieshead]: ../series/head.md
[bodoseriesmap]: ../series/map.md
[bodoseriessortvalues]: ../series/sort_values.md

[overview]: ../index.md/#lazy-evaluation-and-fallback-to-pandas

[bodoseriesstrcapitalize]: https://pandas.pydata.org/docs/reference/api/pandas.Series.str.capitalize.html
[bodoseriesstrcasefold]: https://pandas.pydata.org/docs/reference/api/pandas.Series.str.casefold.html
[bodoseriesstrcenter]: https://pandas.pydata.org/docs/reference/api/pandas.Series.str.center.html
[bodoseriesstrcontains]: https://pandas.pydata.org/docs/reference/api/pandas.Series.str.contains.html
[bodoseriesstrcount]: https://pandas.pydata.org/docs/reference/api/pandas.Series.str.count.html
[bodoseriesstrendswith]: https://pandas.pydata.org/docs/reference/api/pandas.Series.str.endswith.html
[bodoseriesstrfind]: https://pandas.pydata.org/docs/reference/api/pandas.Series.str.find.html
[bodoseriesstrfindall]: https://pandas.pydata.org/docs/reference/api/pandas.Series.str.findall.html
[bodoseriesstrfullmatch]: https://pandas.pydata.org/docs/reference/api/pandas.Series.str.fullmatch.html
[bodoseriesstrget]: https://pandas.pydata.org/docs/reference/api/pandas.Series.str.get.html
[bodoseriesstrindex]: https://pandas.pydata.org/docs/reference/api/pandas.Series.str.index.html
[bodoseriesstrisalnum]: https://pandas.pydata.org/docs/reference/api/pandas.Series.str.isalnum.html
[bodoseriesstrisalpha]: https://pandas.pydata.org/docs/reference/api/pandas.Series.str.isalpha.html
[bodoseriesstrisdecimal]: https://pandas.pydata.org/docs/reference/api/pandas.Series.str.isdecimal.html
[bodoseriesstrisdigit]: https://pandas.pydata.org/docs/reference/api/pandas.Series.str.isdigit.html
[bodoseriesstrislower]: https://pandas.pydata.org/docs/reference/api/pandas.Series.str.islower.html
[bodoseriesstrisnumeric]: https://pandas.pydata.org/docs/reference/api/pandas.Series.str.isnumeric.html
[bodoseriesstrisspace]: https://pandas.pydata.org/docs/reference/api/pandas.Series.str.isspace.html
[bodoseriesstristitle]: https://pandas.pydata.org/docs/reference/api/pandas.Series.str.istitle.html
[bodoseriesstrisupper]: https://pandas.pydata.org/docs/reference/api/pandas.Series.str.isupper.html
[bodoseriesstrlen]: https://pandas.pydata.org/docs/reference/api/pandas.Series.str.len.html
[bodoseriesstrljust]: https://pandas.pydata.org/docs/reference/api/pandas.Series.str.ljust.html
[bodoseriesstrlower]: https://pandas.pydata.org/docs/reference/api/pandas.Series.str.lower.html
[bodoseriesstrlstrip]: https://pandas.pydata.org/docs/reference/api/pandas.Series.str.lstrip.html
[bodoseriesstrmatch]: https://pandas.pydata.org/docs/reference/api/pandas.Series.str.match.html
[bodoseriesstrpad]: https://pandas.pydata.org/docs/reference/api/pandas.Series.str.pad.html
[bodoseriesstrremoveprefix]: https://pandas.pydata.org/docs/reference/api/pandas.Series.str.removeprefix.html
[bodoseriesstrremovesuffix]: https://pandas.pydata.org/docs/reference/api/pandas.Series.str.removesuffix.html
[bodoseriesstrrepeat]: https://pandas.pydata.org/docs/reference/api/pandas.Series.str.repeat.html
[bodoseriesstrreplace]: https://pandas.pydata.org/docs/reference/api/pandas.Series.str.replace.html
[bodoseriesstrrfind]: https://pandas.pydata.org/docs/reference/api/pandas.Series.str.rfind.html
[bodoseriesstrrindex]: https://pandas.pydata.org/docs/reference/api/pandas.Series.str.rindex.html
[bodoseriesstrrjust]: https://pandas.pydata.org/docs/reference/api/pandas.Series.str.rjust.html
[bodoseriesstrrstrip]: https://pandas.pydata.org/docs/reference/api/pandas.Series.str.rstrip.html
[bodoseriesstrslice]: https://pandas.pydata.org/docs/reference/api/pandas.Series.str.slice.html
[bodoseriesstrslicereplace]: https://pandas.pydata.org/docs/reference/api/pandas.Series.str.slice_replace.html
[bodoseriesstrstartswith]: https://pandas.pydata.org/docs/reference/api/pandas.Series.str.startswith.html
[bodoseriesstrstrip]: https://pandas.pydata.org/docs/reference/api/pandas.Series.str.strip.html
[bodoseriesstrswapcase]: https://pandas.pydata.org/docs/reference/api/pandas.Series.str.swapcase.html
[bodoseriesstrtitle]: https://pandas.pydata.org/docs/reference/api/pandas.Series.str.title.html
[bodoseriesstrtranslate]: https://pandas.pydata.org/docs/reference/api/pandas.Series.str.translate.html
[bodoseriesstrupper]: https://pandas.pydata.org/docs/reference/api/pandas.Series.str.upper.html
[bodoseriesstrwrap]: https://pandas.pydata.org/docs/reference/api/pandas.Series.str.wrap.html
[bodoseriesstrzfill]: https://pandas.pydata.org/docs/reference/api/pandas.Series.str.zfill.html


[bodoseriesdtyear]: https://pandas.pydata.org/docs/reference/api/pandas.Series.dt.year.html
[bodoseriesdtmonth]: https://pandas.pydata.org/docs/reference/api/pandas.Series.dt.month.html
[bodoseriesdtday]: https://pandas.pydata.org/docs/reference/api/pandas.Series.dt.day.html
[bodoseriesdthour]: https://pandas.pydata.org/docs/reference/api/pandas.Series.dt.hour.html
[bodoseriesdtminute]: https://pandas.pydata.org/docs/reference/api/pandas.Series.dt.minute.html
[bodoseriesdtsecond]: https://pandas.pydata.org/docs/reference/api/pandas.Series.dt.second.html
[bodoseriesdtmicrosecond]: https://pandas.pydata.org/docs/reference/api/pandas.Series.dt.microsecond.html
[bodoseriesdtnanosecond]: https://pandas.pydata.org/docs/reference/api/pandas.Series.dt.nanosecond.html
[bodoseriesdtdayofweek]: https://pandas.pydata.org/docs/reference/api/pandas.Series.dt.dayofweek.html
[bodoseriesdtday_of_week]: https://pandas.pydata.org/docs/reference/api/pandas.Series.dt.dayofweek.html
[bodoseriesdtweekday]: https://pandas.pydata.org/docs/reference/api/pandas.Series.dt.weekday.html
[bodoseriesdtdayofyear]: https://pandas.pydata.org/docs/reference/api/pandas.Series.dt.dayofyear.html
[bodoseriesdtday_of_year]: https://pandas.pydata.org/docs/reference/api/pandas.Series.dt.dayofyear.html
[bodoseriesdtdays_in_month]: https://pandas.pydata.org/docs/reference/api/pandas.Series.dt.days_in_month.html
[bodoseriesdtquarter]: https://pandas.pydata.org/docs/reference/api/pandas.Series.dt.quarter.html
[bodoseriesdtis_month_start]: https://pandas.pydata.org/docs/reference/api/pandas.Series.dt.is_month_start.html
[bodoseriesdtis_month_end]: https://pandas.pydata.org/docs/reference/api/pandas.Series.dt.is_month_end.html
[bodoseriesdtis_quarter_start]: https://pandas.pydata.org/docs/reference/api/pandas.Series.dt.is_quarter_start.html
[bodoseriesdtis_quarter_end]: https://pandas.pydata.org/docs/reference/api/pandas.Series.dt.is_quarter_end.html
[bodoseriesdtis_year_start]: https://pandas.pydata.org/docs/reference/api/pandas.Series.dt.is_year_start.html
[bodoseriesdtis_year_end]: https://pandas.pydata.org/docs/reference/api/pandas.Series.dt.is_year_end.html
[bodoseriesdtis_leap_year]: https://pandas.pydata.org/docs/reference/api/pandas.Series.dt.is_leap_year.html
[bodoseriesdtdaysinmonth]: https://pandas.pydata.org/docs/reference/api/pandas.Series.dt.daysinmonth.html
[bodoseriesdtdays_in_month]: https://pandas.pydata.org/docs/reference/api/pandas.Series.dt.days_in_month.html
[bodoseriesdtdate]: https://pandas.pydata.org/docs/reference/api/pandas.Series.dt.date.html
[bodoseriesdttime]: https://pandas.pydata.org/docs/reference/api/pandas.Series.dt.time.html

[bodoseriesisin]: https://pandas.pydata.org/docs/reference/api/pandas.Series.isin.html
[bodoseriesnotnull]: https://pandas.pydata.org/docs/reference/api/pandas.Series.notnull.html
[bodoseriesisnull]: https://pandas.pydata.org/docs/reference/api/pandas.Series.isnull.html
[bodoseriesffill]: https://pandas.pydata.org/docs/reference/api/pandas.Series.ffill.html
[bodoseriesbfill]: https://pandas.pydata.org/docs/reference/api/pandas.Series.bfill.html
[bodoseriesreplace]: https://pandas.pydata.org/docs/reference/api/pandas.Series.replace.html
[bodoseriesclip]: https://pandas.pydata.org/docs/reference/api/pandas.Series.clip.html
[bodoseriesabs]: https://pandas.pydata.org/docs/reference/api/pandas.Series.abs.html
[bodoseriesround]: https://pandas.pydata.org/docs/reference/api/pandas.Series.round.html

[bodoseriesdtnormalize]: https://pandas.pydata.org/docs/reference/api/pandas.Series.dt.normalize.html
[bodoseriesdtfloor]: https://pandas.pydata.org/docs/reference/api/pandas.Series.dt.floor.html
[bodoseriesdtceil]: https://pandas.pydata.org/docs/reference/api/pandas.Series.dt.ceil.html
[bodoseriesdtmonth_name]: https://pandas.pydata.org/docs/reference/api/pandas.Series.dt.month_name.html
[bodoseriesdtday_name]: https://pandas.pydata.org/docs/reference/api/pandas.Series.dt.day_name.html
[bodoseriesdtstrftime]: https://pandas.pydata.org/docs/reference/api/pandas.Series.dt.strftime.html
[bodoseriesstrpartition]: https://pandas.pydata.org/docs/reference/api/pandas.Series.str.partition.html
[bodoseriesstrrpartition]: https://pandas.pydata.org/docs/reference/api/pandas.Series.str.rpartition.html
[bodoseriesdtquarter]: https://pandas.pydata.org/docs/reference/api/pandas.Series.dt.quarter.html
[bodoseriesstrnormalize]: https://pandas.pydata.org/docs/reference/api/pandas.Series.str.normalize.html


[bodoseriesstrjoin]: https://pandas.pydata.org/docs/reference/api/pandas.Series.str.join.html
[bodoseriesstrencode]: https://pandas.pydata.org/docs/reference/api/pandas.Series.str.encode.html
[bodoseriesstrdecode]: https://pandas.pydata.org/docs/reference/api/pandas.Series.str.decode.html

[bodoseriesstrcat]: https://pandas.pydata.org/docs/reference/api/pandas.Series.str.cat.html
[bodoseriesstrextract]: https://pandas.pydata.org/docs/reference/api/pandas.Series.str.extract.html
[bodoseriesstrsplit]: https://pandas.pydata.org/docs/reference/api/pandas.Series.str.split.html
[bodoseriesstrrsplit]: https://pandas.pydata.org/docs/reference/api/pandas.Series.str.rsplit.html
[bodoseriesisna]: https://pandas.pydata.org/docs/reference/api/pandas.Series.isna.html
[bodoseriesnotna]: https://pandas.pydata.org/docs/reference/api/pandas.Series.notna.html
[bodoseriesdttotal_seconds]: https://pandas.pydata.org/docs/reference/api/pandas.Series.dt.total_seconds.html

[bodoseriescount]: https://pandas.pydata.org/docs/reference/api/pandas.Series.count.html
[bodoseriesmax]: https://pandas.pydata.org/docs/reference/api/pandas.Series.max.html
[bodoseriesmin]: https://pandas.pydata.org/docs/reference/api/pandas.Series.min.html
[bodoseriesproduct]: https://pandas.pydata.org/docs/reference/api/pandas.Series.product.html
[bodoseriessum]: https://pandas.pydata.org/docs/reference/api/pandas.Series.sum.html

[bodoseriesdtround]: https://pandas.pydata.org/docs/reference/api/pandas.Series.dt.round.html
[bodoseriesdttz_localize]: https://pandas.pydata.org/docs/reference/api/pandas.Series.dt.tz_localize.html


[bodoseriesdtdays]: https://pandas.pydata.org/docs/reference/api/pandas.Series.dt.days.html
[bodoseriesdtseconds]: https://pandas.pydata.org/docs/reference/api/pandas.Series.dt.seconds.html
[bodoseriesdtmicroseconds]: https://pandas.pydata.org/docs/reference/api/pandas.Series.dt.microseconds.html
[bodoseriesdtnanoseconds]: https://pandas.pydata.org/docs/reference/api/pandas.Series.dt.nanoseconds.html

[bodoseriesdtcomponents]: https://pandas.pydata.org/docs/reference/api/pandas.Series.dt.components.html
[bodoseriesdtisocalendar]: https://pandas.pydata.org/docs/reference/api/pandas.Series.dt.isocalendar.html

[bodoseriesagg]: https://pandas.pydata.org/docs/reference/api/pandas.Series.agg.html
[bodoseriesaggregate]: https://pandas.pydata.org/docs/reference/api/pandas.Series.aggregate.html
