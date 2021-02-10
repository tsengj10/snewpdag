run: init
	echo python -m snewpdag # needs config and input JSON filenames

lightcurvesim:
	cd externals/lightcurve_match && make simulation
	cd externals/lightcurve_match/matching && make getdelay

test: lightcurvesim
	python -m unittest snewpdag.tests.test_basic_node
	python -m unittest snewpdag.tests.test_inputs
	python -m unittest snewpdag.tests.test_app
	python -m unittest snewpdag.tests.test_combinemaps
	python -m unittest snewpdag.tests.test_timedistdiff

init:
	pip install -r requirements.txt

.PHONY: init run test
