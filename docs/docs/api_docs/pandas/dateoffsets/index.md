# Date Offsets

Bodo supports a subset of the offset types in `pandas.tseries.offsets`:

## DateOffset

- \[`pd.tseries.offsets.DateOffset`\][pdtseriesoffsetsdateoffset]
- \[`pd.tseries.offsets.MonthBegin`\][pdtseriesoffsetsmonthbegin]
- \[`pd.tseries.offsets.MonthEnd`\][pdtseriesoffsetsmonthend]
- \[`pd.tseries.offsets.Week`\][pdtseriesoffsetsweek]

### Properties

- [pd.tseries.offsets.DateOffset.normalize\`][pdtseriesoffsetsdateoffsetnormalize]
- \[`pd.tseries.offsets.DateOffset.n`\][pdtseriesoffsetsdateoffsetn]

## Binary Operations

For all offsets, addition and subtraction with a scalar
`datetime.date`, `datetime.datetime` or `pandas.Timestamp`
is supported. Multiplication is also supported with a scalar integer.
