# Daffodil Roadmap: ROADMAP.md

<pre>
    TODO
        Add ability to provide active tables with column sorting in HTML output derived from MD table spec.
    
        SQL extension Plan:
            create base class for core operations.
            use this to create daf_sql class to handle sql functionality.
            extend daffodil methods to handle sql tables with similar syntax.
    
        Consider method adjust_cols(select_cols, drop_cols, add_cols_dtypes, default_val) which would make a single pass through the
            table to select (keep), drop, or add columns. This can improve performance by avoiding creating a new
            table and potentially nearly doubling the data. Strategy will be to apply() in place so the table does 
            not grow in size, and this mimics the behavior in SQL tables.
    
        offer dropping unexpected columns when doing concat/append ??
            keys mismatch: daf: (['ballot_id', 'style_num', 'precinct', 'contest', 'option', 'has_indication', 'num_marks', 'num_votes', 'pixel_metric_value', 'sm_pmv', 'writein_name', 'overvotes', 'undervotes', 'ssidx', 'delta_y', 'ev_coord_str', 'ev_precinct_id', 'target_pixels', 'gray_eval', 'is_bmd', 'wipmv', 'p', 'x', 'y', 'w', 'h', 'b', 'bmd_str'])
            other_instance:     (['ballot_id', 'style_num', 'precinct', 'contest', 'option', 'has_indication', 'num_marks', 'num_votes', 'pixel_metric_value', 'sm_pmv', 'writein_name', 'overvotes', 'undervotes', 'ssidx', 'delta_y', 'ev_coord_str', 'ev_precinct_id', 'target_pixels', 'gray_eval', 'is_bmd', 'wipmv', 'p', 'x', 'y', 'w', 'h'])
            > daf.py(2003)concat()
    
            No: It will probably be better to keep a value of the num cols and not calculate every time. (not worth it)
            No: add use of pickledjson to express contents of individual cells when not jsonable. (Use Pyon)
            tests: need to add
                __str__ and __repr__
                
                apply_dtypes()
                from_csv_buff()
                from_numpy()
                to_csv_buff()
                append()  
                    empty data item
                    simple list with columns defined.
                    no columns defined, just append to lol.
                concat()
                    not other instance
                    empty current daf
                    self.hd != other_instance.hd
                extend()
                    no input records
                    some records empty
                record_append()
                    empty record
                    overwrite of existing
                select_records_daf   (remove?)
                    no keyfield
                    inverse
                        len(keys_ls) > 10 and len(self.kd) > 30
                        smaller cases
                iloc                (remove?)
                    no hd
                icol_to_la          (remove?)
                    unique
                    omit_nulls
                assign_record       (remove?)
                    no keyfield defined.
                update_by_keylist <-- remove?
                insert_irow()
                set_col_irows() <-- remove.
                set_icol()  <-- remove.
                set_icol_irows() <-- remove.
                find_replace
                sort_by_colname(
                apply_formulas()
                    no self
                    formula shape differs
                    error in formula
                apply
                    keylist without keyfield
                update_row()
                apply_in_place()
                    keylist without keyfield
                reduce 
                    by == 'col'
                manifest_apply()
                manifest_reduce()
                manifest_process()
                groupby()
                groupby_reduce()
                daf_valuecount()
                groupsum_daf()
                set_col2_from_col1_using_regex_select
                    not col2
                apply_to_col
                sum_da 
                    one col
                    cols_list > 10
                count_values_da()
                sum_dodis -- needed?
                valuecounts_for_colname
                    omit_nulls
                valuecounts_for_colnames_ls_selectedby_colname
                    not colnames_ls
                gen_stats_daf()
                to_md()
                dict_to_md() <-- needed?
                value_counts_daf()
                
            daf_utils
                is_linux()
                apply_dtypes_to_hdlol()     DEPRECATED
                    empty case
                select_col_of_lol_by_col_idx()
                    col_idx out of range.
                json_encode  --> now 'to_json()'
                make_strbool <-- remove? yes, remove.
                test_strbool >-- remove? yes, remove.
                xlsx_to_csv
                add_trailing_columns_csv()
                insert_col_in_lol_at_icol()
                    col_la empty
                insert_row_in_lol_at_irow() unused?
                calc_chunk_sizes
                    not num_items or not max_chunk_size
                sort_lol_by_col
                set_dict_dtypes 
                    various cases
                list_stats
                    REMOVE?
                list_stats_index
                list_stats_attrib
                list_stats_filepaths
                list_stats_scalar
                list_stats_localidx
                is_list_allints
                profile_ls_to_loti
                profile_ls_to_lr
                reduce_lol_cols
                s3path_to_url
                parse_s3path
                transpose_lol
                safe_get_idx
                shorten_str_keeping_ends
                smart_fmt
                str2bool
                safe_del_key
                dod_to_lod
                lod_to_dod
                safe_eval
                safe_min
                safe_stdev
                safe_mean
                sts
                split_dups_list
                clean_numeric_str
                is_numeric
            daf_md
                not tested at all.
                
<pre>