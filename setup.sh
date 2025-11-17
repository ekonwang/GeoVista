
cd $(dirname $0)

mkdir -p .temp
git submodule update --init --recursive

pip install -e .
bash external/gpt-researcher-tool/search_setup_minimal.sh

bash install_geovista.sh
