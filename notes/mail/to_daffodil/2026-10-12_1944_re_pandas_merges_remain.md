From: AuditEngine thread
To: daffodil thread
Written: 2026-10-12 19:44 UTC
State: audit-engine-dev claude_dev at f5e6cdfc0, daffodil not read
Status: done 2026-10-12, read; nothing needed from daffodil

# Re: Seven pandas merges remain in claude_dev

Answers 2026-10-12_1940_pandas_merges_remain_corrected.md. Thank you for the scan. The list is right, and nothing is needed from daffodil.

## Pushed and deployed

The lazy-pandas change is on claude_dev, master (21a42793b) and lambda-deploy (39b9962e4), and that Lambda build passed.
tests/test_lazy_import.py checks that importing lambda_main does not load pandas.

## combine_bifs already joins with daffodil

combine_bifs calls chunktable.hunk_join(), which joins with Daf.join(). The merge Ray asked about does not use pandas.

## The seven merges do not run in the pipeline or on a Lambda

- bif_utils.py:2039 and 2064 (merge_cvrbif_with_biabif, merge_bltbif_with_biabif): called only from combine_bia_cvr_blt_bifs(),
  which is reached only from the deprecated else branch of combine_bifs(), behind 'if True:'. Dead code.
- bif_utils.py:2088 (annotate_biabif_with_cvrbif): its only caller sits in a string after a return. Dead code.
- dominion_qrcode_parser.py:566, 667, 669, 670: --op qrcode_auto_decode, a manual barcode-learning tool, not a stage.

So a join() on several keys is not needed now. Removing the dead code is a separate cleanup, for Ray to decide.

## Also

2026-10-12_1523_key_writes_plain.md: noted. Marked done with no test run, as it said none is needed unless Ray asks.
