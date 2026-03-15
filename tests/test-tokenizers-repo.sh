#!/usr/bin/env bash

if [ $# -lt 2 ]; then
    printf "Usage: $0 <git-repo> <target-folder> [<test-exe>]\n"
    exit 1
fi

if [ $# -eq 3 ]; then
    toktest=$3
else
    toktest="./test-tokenizer-0"
fi

if [ ! -x $toktest ]; then
    printf "Test executable \"$toktest\" not found!\n"
    exit 1
fi

repo=$1
folder=$2

pointer_file() {
    local file=$1
    if [ ! -f "$file" ]; then
        return 1
    fi

    local first_line
    first_line=$(head -n 1 "$file" 2>/dev/null || true)
    [ "$first_line" = "version https://git-lfs.github.com/spec/v1" ]
}

download_hf_lfs_fallback() {
    local repo_url=$1
    local root=$2
    local file=$3

    case "$repo_url" in
        https://huggingface.co/*)
            ;;
        *)
            return 1
            ;;
    esac

    local relative_path=${file#"$root"/}
    local resolve_url="${repo_url%/}/resolve/main/${relative_path}"
    local tmp_file="${file}.download"

    printf "Downloading LFS payload for \"%s\"...\n" "$relative_path"
    if curl -fsSL --retry 3 --output "$tmp_file" "$resolve_url"; then
        mv "$tmp_file" "$file"
        return 0
    fi

    rm -f "$tmp_file"
    return 1
}

hydrate_lfs_payloads() {
    local repo_url=$1
    local root=$2
    local has_pointers=0
    local gguf

    while IFS= read -r gguf; do
        if pointer_file "$gguf"; then
            has_pointers=1
            if git lfs version >/dev/null 2>&1; then
                break
            fi
            download_hf_lfs_fallback "$repo_url" "$root" "$gguf" || return 1
        fi
    done < <(find "$root" -type f -name '*.gguf' | sort)

    if [ "$has_pointers" -eq 1 ] && git lfs version >/dev/null 2>&1; then
        (cd "$root" && git lfs pull) || return 1
    fi

    return 0
}

if [ -d $folder ] && [ -d $folder/.git ]; then
    (cd $folder; git pull)
else
    git clone $repo $folder

    # byteswap models if on big endian
    if [ "$(uname -m)" = s390x ]; then
        for f in $folder/*/*.gguf; do
            echo YES | python3 "$(dirname $0)/../gguf-py/gguf/scripts/gguf_convert_endian.py" $f big
        done
    fi
fi

hydrate_lfs_payloads "$repo" "$folder"

find "$folder" -type f -name '*.gguf' | sort | while IFS= read -r gguf; do
    if [ -f $gguf.inp ] && [ -f $gguf.out ]; then
        $toktest $gguf
    else
        printf "Found \"$gguf\" without matching inp/out files, ignoring...\n"
    fi
done
