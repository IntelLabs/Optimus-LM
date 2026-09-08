import os
import sys

info_dict = {
    "algebraic-stack" : 
        {
            "base_url" : "https://huggingface.co/datasets/allenai/OLMoE-mix-0924/resolve/main/data/algebraic-stack/",
            "groups" : 
                [
                    {"start_id" : 0, "end_id" :  15, "zfill_width" : 4, "prefix" : "algebraic-stack-train-", "extension" : ".json.gz"}
                ]
        },
    "dclm" : 
        {
            "base_url" : "https://huggingface.co/datasets/allenai/OLMoE-mix-0924/resolve/main/data/dclm/",
            "groups" : 
                [
                    {"start_id" : 0, "end_id" :  1969, "zfill_width" : 4, "prefix" : "dclm-", "extension" : ".json.zst"}
                ]
        },
    "open-web-math" : 
        {
            "base_url" : "https://huggingface.co/datasets/allenai/OLMoE-mix-0924/resolve/main/data/open-web-math/",
            "groups" : 
                [
                    {"start_id" : 41, "end_id" :  53, "zfill_width" : 3, "prefix" : "", "extension" : ".jsonl.gz"},
                    {"start_id" :  0, "end_id" :  12, "zfill_width" : 4, "prefix" : "open-web-math-train-", "extension" : ".json.gz"}
                ]
        },
    "pes2o" : 
        {
            "base_url" : "https://huggingface.co/datasets/allenai/OLMoE-mix-0924/resolve/main/data/pes2o/",
            "groups" : 
                [
                    {"start_id" : 0, "end_id" :  25, "zfill_width" : 4, "prefix" : "pes2o-", "extension" : ".json.gz"}
                ]
        },
    "starcoder" : 
        {
            "base_url" : "https://huggingface.co/datasets/allenai/OLMoE-mix-0924/resolve/main/data/starcoder/",
            "groups" : 
                [
                    {"start_id" : 0, "end_id" :  0, "zfill_width" : 4, "extension" : ".json.gz", "prefix" : "ada-"},
                    {"start_id" : 0, "end_id" :  0, "zfill_width" : 4, "extension" : ".json.gz", "prefix" : "agda-"},
                    {"start_id" : 0, "end_id" :  0, "zfill_width" : 4, "extension" : ".json.gz", "prefix" : "alloy-"},
                    {"start_id" : 0, "end_id" :  0, "zfill_width" : 4, "extension" : ".json.gz", "prefix" : "antlr-"},
                    {"start_id" : 0, "end_id" :  0, "zfill_width" : 4, "extension" : ".json.gz", "prefix" : "applescript-"},
                    {"start_id" : 0, "end_id" :  1, "zfill_width" : 4, "extension" : ".json.gz", "prefix" : "assembly-"},
                    {"start_id" : 0, "end_id" :  0, "zfill_width" : 4, "extension" : ".json.gz", "prefix" : "augeas-"},
                    {"start_id" : 0, "end_id" :  0, "zfill_width" : 4, "extension" : ".json.gz", "prefix" : "awk-"},
                    {"start_id" : 0, "end_id" :  0, "zfill_width" : 4, "extension" : ".json.gz", "prefix" : "batchfile-"},
                    {"start_id" : 0, "end_id" :  0, "zfill_width" : 4, "extension" : ".json.gz", "prefix" : "bluespec-"},
                    {"start_id" : 0, "end_id" : 52, "zfill_width" : 4, "extension" : ".json.gz", "prefix" : "c-"},
                    {"start_id" : 0, "end_id" : 44, "zfill_width" : 4, "extension" : ".json.gz", "prefix" : "c-sharp-"},
                    {"start_id" : 0, "end_id" :  0, "zfill_width" : 4, "extension" : ".json.gz", "prefix" : "clojure-"},
                    {"start_id" : 0, "end_id" :  0, "zfill_width" : 4, "extension" : ".json.gz", "prefix" : "cmake-"},
                    {"start_id" : 0, "end_id" :  0, "zfill_width" : 4, "extension" : ".json.gz", "prefix" : "coffeescript-"},
                    {"start_id" : 0, "end_id" :  1, "zfill_width" : 4, "extension" : ".json.gz", "prefix" : "common-lisp-"},
                    {"start_id" : 0, "end_id" : 47, "zfill_width" : 4, "extension" : ".json.gz", "prefix" : "cpp-"},
                    {"start_id" : 0, "end_id" : 11, "zfill_width" : 4, "extension" : ".json.gz", "prefix" : "css-"},
                    {"start_id" : 0, "end_id" :  0, "zfill_width" : 4, "extension" : ".json.gz", "prefix" : "cuda-"},
                    {"start_id" : 0, "end_id" :  3, "zfill_width" : 4, "extension" : ".json.gz", "prefix" : "dart-"},
                    {"start_id" : 0, "end_id" :  0, "zfill_width" : 4, "extension" : ".json.gz", "prefix" : "dockerfile-"},
                    {"start_id" : 0, "end_id" :  0, "zfill_width" : 4, "extension" : ".json.gz", "prefix" : "elixir-"},
                    {"start_id" : 0, "end_id" :  0, "zfill_width" : 4, "extension" : ".json.gz", "prefix" : "elm-"},
                    {"start_id" : 0, "end_id" :  0, "zfill_width" : 4, "extension" : ".json.gz", "prefix" : "emacs-lisp-"},
                    {"start_id" : 0, "end_id" :  0, "zfill_width" : 4, "extension" : ".json.gz", "prefix" : "erlang-"},
                    {"start_id" : 0, "end_id" :  0, "zfill_width" : 4, "extension" : ".json.gz", "prefix" : "f-sharp-"},
                    {"start_id" : 0, "end_id" :  1, "zfill_width" : 4, "extension" : ".json.gz", "prefix" : "fortran-"},
                    {"start_id" : 0, "end_id" : 54, "zfill_width" : 4, "extension" : ".json.gz", "prefix" : "git-commits-cleaned-"},
                    {"start_id" : 0, "end_id" : 58, "zfill_width" : 4, "extension" : ".json.gz", "prefix" : "github-issues-filtered-structured-"},
                    {"start_id" : 0, "end_id" :  0, "zfill_width" : 4, "extension" : ".json.gz", "prefix" : "glsl-"},
                    {"start_id" : 0, "end_id" : 23, "zfill_width" : 4, "extension" : ".json.gz", "prefix" : "go-"},
                    {"start_id" : 0, "end_id" :  0, "zfill_width" : 4, "extension" : ".json.gz", "prefix" : "groovy-"},
                    {"start_id" : 0, "end_id" :  2, "zfill_width" : 4, "extension" : ".json.gz", "prefix" : "haskell-"},
                    {"start_id" : 0, "end_id" : 28, "zfill_width" : 4, "extension" : ".json.gz", "prefix" : "html-"},
                    {"start_id" : 0, "end_id" :  0, "zfill_width" : 4, "extension" : ".json.gz", "prefix" : "idris-"},
                    {"start_id" : 0, "end_id" :  0, "zfill_width" : 4, "extension" : ".json.gz", "prefix" : "isabelle-"},
                    {"start_id" : 0, "end_id" : 86, "zfill_width" : 4, "extension" : ".json.gz", "prefix" : "java-"},
                    {"start_id" : 0, "end_id" :  0, "zfill_width" : 4, "extension" : ".json.gz", "prefix" : "java-server-pages-"},
                    {"start_id" : 0, "end_id" : 64, "zfill_width" : 4, "extension" : ".json.gz", "prefix" : "javascript-"},
                    {"start_id" : 0, "end_id" :  5, "zfill_width" : 4, "extension" : ".json.gz", "prefix" : "json-"},
                    {"start_id" : 0, "end_id" :  1, "zfill_width" : 4, "extension" : ".json.gz", "prefix" : "julia-"},
                    {"start_id" : 0, "end_id" :  7, "zfill_width" : 4, "extension" : ".json.gz", "prefix" : "jupyter-scripts-dedup-filtered-"},
                    {"start_id" : 0, "end_id" :  5, "zfill_width" : 4, "extension" : ".json.gz", "prefix" : "jupyter-structured-clean-dedup-"},
                    {"start_id" : 0, "end_id" :  5, "zfill_width" : 4, "extension" : ".json.gz", "prefix" : "kotlin-"},
                    {"start_id" : 0, "end_id" :  0, "zfill_width" : 4, "extension" : ".json.gz", "prefix" : "lean-"},
                    {"start_id" : 0, "end_id" :  0, "zfill_width" : 4, "extension" : ".json.gz", "prefix" : "literate-agda-"},
                    {"start_id" : 0, "end_id" :  0, "zfill_width" : 4, "extension" : ".json.gz", "prefix" : "literate-coffeescript-"},
                    {"start_id" : 0, "end_id" :  0, "zfill_width" : 4, "extension" : ".json.gz", "prefix" : "literate-haskell-"},
                    {"start_id" : 0, "end_id" :  2, "zfill_width" : 4, "extension" : ".json.gz", "prefix" : "lua-"},
                    {"start_id" : 0, "end_id" :  1, "zfill_width" : 4, "extension" : ".json.gz", "prefix" : "makefile-"},
                    {"start_id" : 0, "end_id" :  0, "zfill_width" : 4, "extension" : ".json.gz", "prefix" : "maple-"},
                    {"start_id" : 0, "end_id" : 78, "zfill_width" : 4, "extension" : ".json.gz", "prefix" : "markdown-"},
                    {"start_id" : 0, "end_id" :  1, "zfill_width" : 4, "extension" : ".json.gz", "prefix" : "mathematica-"},
                    {"start_id" : 0, "end_id" :  0, "zfill_width" : 4, "extension" : ".json.gz", "prefix" : "matlab-"},
                    {"start_id" : 0, "end_id" :  0, "zfill_width" : 4, "extension" : ".json.gz", "prefix" : "ocaml-"},
                    {"start_id" : 0, "end_id" :  1, "zfill_width" : 4, "extension" : ".json.gz", "prefix" : "pascal-"},
                    {"start_id" : 0, "end_id" :  2, "zfill_width" : 4, "extension" : ".json.gz", "prefix" : "perl-"},
                    {"start_id" : 0, "end_id" : 60, "zfill_width" : 4, "extension" : ".json.gz", "prefix" : "php-"},
                    {"start_id" : 0, "end_id" :  1, "zfill_width" : 4, "extension" : ".json.gz", "prefix" : "powershell-"},
                    {"start_id" : 0, "end_id" :  0, "zfill_width" : 4, "extension" : ".json.gz", "prefix" : "prolog-"},
                    {"start_id" : 0, "end_id" :  0, "zfill_width" : 4, "extension" : ".json.gz", "prefix" : "protocol-buffer-"},
                    {"start_id" : 0, "end_id" : 58, "zfill_width" : 4, "extension" : ".json.gz", "prefix" : "python-"},
                    {"start_id" : 0, "end_id" :  0, "zfill_width" : 4, "extension" : ".json.gz", "prefix" : "r-"},
                    {"start_id" : 0, "end_id" :  0, "zfill_width" : 4, "extension" : ".json.gz", "prefix" : "racket-"},
                    {"start_id" : 0, "end_id" :  3, "zfill_width" : 4, "extension" : ".json.gz", "prefix" : "restructuredtext-"},
                    {"start_id" : 0, "end_id" :  0, "zfill_width" : 4, "extension" : ".json.gz", "prefix" : "rmarkdown-"},
                    {"start_id" : 0, "end_id" :  6, "zfill_width" : 4, "extension" : ".json.gz", "prefix" : "ruby-"},
                    {"start_id" : 0, "end_id" :  8, "zfill_width" : 4, "extension" : ".json.gz", "prefix" : "rust-"},
                    {"start_id" : 0, "end_id" :  0, "zfill_width" : 4, "extension" : ".json.gz", "prefix" : "sas-"},
                    {"start_id" : 0, "end_id" :  4, "zfill_width" : 4, "extension" : ".json.gz", "prefix" : "scala-"},
                    {"start_id" : 0, "end_id" :  0, "zfill_width" : 4, "extension" : ".json.gz", "prefix" : "scheme-"},
                    {"start_id" : 0, "end_id" :  3, "zfill_width" : 4, "extension" : ".json.gz", "prefix" : "shell-"},
                    {"start_id" : 0, "end_id" :  0, "zfill_width" : 4, "extension" : ".json.gz", "prefix" : "smalltalk-"},
                    {"start_id" : 0, "end_id" :  0, "zfill_width" : 4, "extension" : ".json.gz", "prefix" : "solidity-"},
                    {"start_id" : 0, "end_id" :  0, "zfill_width" : 4, "extension" : ".json.gz", "prefix" : "sparql-"},
                    {"start_id" : 0, "end_id" : 10, "zfill_width" : 4, "extension" : ".json.gz", "prefix" : "sql-"},
                    {"start_id" : 0, "end_id" :  0, "zfill_width" : 4, "extension" : ".json.gz", "prefix" : "stan-"},
                    {"start_id" : 0, "end_id" :  0, "zfill_width" : 4, "extension" : ".json.gz", "prefix" : "standard-ml-"},
                    {"start_id" : 0, "end_id" :  0, "zfill_width" : 4, "extension" : ".json.gz", "prefix" : "stata-"},
                    {"start_id" : 0, "end_id" :  0, "zfill_width" : 4, "extension" : ".json.gz", "prefix" : "systemverilog-"},
                    {"start_id" : 0, "end_id" :  0, "zfill_width" : 4, "extension" : ".json.gz", "prefix" : "tcl-"},
                    {"start_id" : 0, "end_id" :  0, "zfill_width" : 4, "extension" : ".json.gz", "prefix" : "tcsh-"},
                    {"start_id" : 0, "end_id" :  5, "zfill_width" : 4, "extension" : ".json.gz", "prefix" : "tex-"},
                    {"start_id" : 0, "end_id" :  0, "zfill_width" : 4, "extension" : ".json.gz", "prefix" : "thrift-"},
                    {"start_id" : 0, "end_id" : 26, "zfill_width" : 4, "extension" : ".json.gz", "prefix" : "typescript-"},
                    {"start_id" : 0, "end_id" :  0, "zfill_width" : 4, "extension" : ".json.gz", "prefix" : "verilog-"},
                    {"start_id" : 0, "end_id" :  0, "zfill_width" : 4, "extension" : ".json.gz", "prefix" : "vhdl-"},
                    {"start_id" : 0, "end_id" :  1, "zfill_width" : 4, "extension" : ".json.gz", "prefix" : "visual-basic-"},
                    {"start_id" : 0, "end_id" :  0, "zfill_width" : 4, "extension" : ".json.gz", "prefix" : "xslt-"},
                    {"start_id" : 0, "end_id" :  0, "zfill_width" : 4, "extension" : ".json.gz", "prefix" : "yacc-"},
                    {"start_id" : 0, "end_id" :  3, "zfill_width" : 4, "extension" : ".json.gz", "prefix" : "yaml-"},
                    {"start_id" : 0, "end_id" :  0, "zfill_width" : 4, "extension" : ".json.gz", "prefix" : "zig-"}
                ]
        },
    "wiki" : {
        "base_url" : "https://huggingface.co/datasets/allenai/OLMoE-mix-0924/resolve/main/data/wiki/",
        "groups" :
            [
                {"start_id" : 0, "end_id" :  1, "zfill_width" : 4, "prefix" : "wiki-", "extension" : ".json.gz"}
            ]
    }
}

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python generate_urls.py <dataset_path>")
        sys.exit(1)

    dataset_path = sys.argv[1]

    global_urls_file_path = os.path.join(dataset_path, "urls.txt")
    global_file = open(global_urls_file_path, "w")

    # Generate urls for sub datasets
    for dataset in info_dict:
        base_url = info_dict[dataset]["base_url"]
        groups = info_dict[dataset]["groups"]

        file_path = os.path.join(dataset_path, f"urls_{dataset}.txt")
        file = open(file_path, "w")
        for group in groups:
            for i in range(group["start_id"], group["end_id"] + 1):
                # The number is formated to be zero-padded to 4 digits
                id = str(i).zfill(group["zfill_width"])
                url = f"{base_url}{group['prefix']}{id}{group['extension']}"
                file.write(url+"\n") # Writing to the local file
                global_file.write(url+"\n") # Writing to the global file as well
        file.close()
    global_file.close()