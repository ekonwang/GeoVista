
cd $(dirname $0)
mkdir -p .temp

# for basic requirements
bash install_geovista.sh

# for verl (reinforcement learning)
pip install -e .

# for web search infra
git submodule update --init --recursive
bash external/gpt-researcher-tool/search_setup_minimal.sh
