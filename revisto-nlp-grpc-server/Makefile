# Directories
PROTO_DIR := proto
OUT_DIR := generated
PYTHONIC_STUBS_DIR := pythonic_stubs

# Python script to auto-generate files for *.proto files and fix imports
FIX_IMPORT_SCRIPT := generate_protos.py

# BetterProto generation
BP_CMD := python -m grpc_tools.protoc \
	-I $(PROTO_DIR) \
	--python_betterproto2_out=$(OUT_DIR)/$(PYTHONIC_STUBS_DIR) \
	--python_betterproto2_opt=client_generation=async \
	$(PROTO_DIR)/*.proto

# ------------------------------------------------------------
# 1. Execute Python script to create default auto-generated files with fixed imports
protoc-generate:
	@echo "Running import fixer script..."
	python $(FIX_IMPORT_SCRIPT) $(OUT_DIR) $(PROTO_DIR)

# 2. Generate Pythonic BetterProto async stubs
generate-pythonic-stubs:
	@echo "Creating output folder if it doesn't exist..."
	mkdir -p $(OUT_DIR)/$(PYTHONIC_STUBS_DIR)
	@echo "Generating BetterProto async stubs..."
	$(BP_CMD)

# 3. Run both: generate stubs then fix imports
generate-all: generate-pythonic-stubs protoc-generate
	@echo "All generation complete!"
