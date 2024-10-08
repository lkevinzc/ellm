SHELL=/bin/bash
PROJECT_NAME=ellm
PROJECT_PATH=ellm/
LINT_PATHS=${PROJECT_PATH} tests/ experiment/ scripts/

check_install = python3 -c "import $(1)" || pip3 install $(1) --upgrade
check_install_extra = python3 -c "import $(1)" || pip3 install $(2) --upgrade

test:
	$(call check_install, pytest)
	pytest -s

lint:
	$(call check_install, isort)
	$(call check_install, pylint)
	$(call check_install, mypy)
	isort --check --diff --project=${LINT_PATHS}
	pylint -j 8 --recursive=y ${LINT_PATHS}
	mypy ${PROJECT_PATH}

format:
	$(call check_install, autoflake)
	autoflake --remove-all-unused-imports -i -r ${LINT_PATHS}
	$(call check_install, black)
	black ${LINT_PATHS}
	$(call check_install, isort)
	isort ${LINT_PATHS}

check-docstyle:
	$(call check_install, pydocstyle)
	pydocstyle ${PROJECT_PATH} --convention=google

checks: lint check-docstyle

.PHONY: format lint check-docstyle checks