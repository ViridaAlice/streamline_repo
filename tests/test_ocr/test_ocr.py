import pytest
import sys
import os
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../../ocr_evaluation")
from ocr_evaluation.ocr_eval import main, check_args_validity
from argparse import Namespace

def test_One():
    args_var = {"gt_file":"tests/test_ocr/dummy_gt.jsonl",
                "tess_ocr_folder":"tests/test_ocr/dummy_tess_ocr",
                "output_file":"tests/test_ocr/errTypes/",
                "errorOutput_file":"tests/test_ocr/error_output.txt",
                "bbox_acc":0.5,
                "word_acc":0.95,
                "lit_list":None}
    ns = Namespace(**args_var)
    main(ns)

def test_invalid_bbox_acc():
    args_var = {"gt_file":"tests/test_ocr/dummy_gt.jsonl",
                "tess_ocr_folder":"tests/test_ocr/dummy_tess_ocr",
                "output_file":"tests/test_ocr/errTypes/entire_gt.txt",
                "errorOutput_file":"tests/test_ocr/error_output.txt",
                "bbox_acc":1.5,
                "word_acc":0.95,
                "lit_list":None}
    ns = Namespace(**args_var)
    with pytest.raises(AssertionError):
        check_args_validity(ns)

def test_invalid_word_acc():
    args_var = {"gt_file":"tests/test_ocr/dummy_gt.jsonl",
                "tess_ocr_folder":"tests/test_ocr/dummy_tess_ocr",
                "output_file":"tests/test_ocr/errTypes/entire_gt.txt",
                "bbox_acc":0.1,
                "word_acc":-0.2,
                "lit_list":None}
    ns = Namespace(**args_var)
    with pytest.raises(AssertionError):
        check_args_validity(ns)

def test_valid_lits():
    args_var = {"gt_file":"tests/test_ocr/dummy_gt.jsonl",
                "tess_ocr_folder":"tests/test_ocr/dummy_tess_ocr",
                "output_file":"tests/test_ocr/errTypes/entire_gt.txt",
                "errorOutput_file":"tests/test_ocr/error_output.txt",
                "bbox_acc":0.1,
                "word_acc":0.2,
                "lit_list":["lit18631", "lit20043", "lit22705"]}
    ns = Namespace(**args_var)
    check_args_validity(ns)
    
def test_invalid_lits():
    args_var = {"gt_file":"tests/test_ocr/dummy_gt.jsonl",
                "tess_ocr_folder":"tests/test_ocr/dummy_tess_ocr",
                "output_file":"tests/test_ocr/errTypes/entire_gt.txt",
                "errorOutput_file":"tests/test_ocr/error_output.txt",
                "bbox_acc":0.1,
                "word_acc":0.2,
                "lit_list":["lit186131", "lit20043", "lit2205"]}
    ns = Namespace(**args_var)
    with pytest.raises(AssertionError):
        check_args_validity(ns)

if __name__ == "__main__":
    test_One()
    test_invalid_bbox_acc()
    test_invalid_word_acc()
    test_valid_lits()
    test_invalid_lits()