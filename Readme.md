# BSS-2 Experiment Template

This repository defines a template for experiments conducted on the BrainScaleS-2 hardware platform.

Using the presented structure allows for the collection of

* Python experiment code running on the host computer
* C++ experiment code running on the host computer
* C++ experiment code running on the embedded plasticity processor

in a single repository.

## Getting started

1. Clone this repository
1. Modify `EXPERIMENT_NAME` in the [`wscript`](wscript)
1. Rename the library folders in
   * `src/cc`
   * `src/ppu`
   * `src/py`
   * `tests/**/cc`
   * `tests/**/ppu`
   * `tests/**/py`
1. Replace `template-experiment-dls` with your experiment name in [`.ci/Jenkinsfile`](.ci/Jenkinsfile)
1. If you intend to use Code Review, update [`.gitreview`](.gitreview), else delete it


## Initial Setup

1. Commit the changes you've made
1. Ask the appropriate people for a new repository that will contain your experiment
1. Push your changes to the new repository
1. Add your experiment to the [`repo_db.json`](https://openproject.bioai.eu/projects/projects/repository/revisions/master/changes/repo_db.json)
1. Get [*symwaf2ic*](https://github.com/electronicvisions/waf) (on visionary infrastructure, you can `module load waf`)
1. Create an experiment toplevel and set up your experiment and dependencies
```bash
mkdir my_experiment && cd my_experiment
waf setup --project=my_experiment
```

## Build your Experiment

Using the latest [singularity container](https://openproject.bioai.eu/containers/), run
```bash
singularity exec --app visionary-dls /containers/stable/latest waf configure
singularity exec --app visionary-dls /containers/stable/latest waf build
singularity exec --app visionary-dls /containers/stable/latest waf install
```

## Add a CI Job

Follow the instructions in the [Visionary CI Wiki](https://openproject.bioai.eu/projects/jenlib/wiki/wiki-visionary-ci#job-structure) on how to add a Jenkins job for your project.
