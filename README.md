# README
## What is this repository for?

* *lightblue* is a Japanese CCG parser with DTS representations.  
* Current version: 0.1.6.0.  
* Copyright owner: Daisuke Bekki


## Installing lightblue

### Prerequisite: Haskell Platform
  1. GHC (>= version 7.4)/Cabal (>= version 1.20)
    * Linux: [The Glasgow Haskell Compiler](https://www.haskell.org/ghc/) 
    * Mac: [Haskell for Mac OS X](https://ghcformacosx.github.io/)

In Debian, you may just do it by `sudo apt-get install haskell-platform`.

After installing GHC, update cabal.
```
$ cabal update
```

Installing Haskell-mode for Emacs will help.
```
$ sudo apt-get install haskell-mode
```

### Prerequisite: command-line tools
The followint tools must be installed before executing *lightblue*.

  1. [JUMAN (a User-Extensible Morphological Analyzer for Japanese)](http://nlp.ist.i.kyoto-u.ac.jp/EN/index.php?JUMAN) (>= version 7.0)
  1. tidy (only for prettyprinting XML outputs)
    * Debian: `sudo apt-get install tidy`
    * Mac: `sudo port install tidy`

### Download
Do the following in the directory under which you'd like to install *lightblue*.
```
$ git clone git@bitbucket.org:DaisukeBekki/lightblue.git
```
This operation will create the directory *lightblue* (henceforth we will refer to this directory as <lightblue>) under the directory in which you did the above.

### Configuration and Installation
First you need to add the environment variable LIGHTBLUE and set its value as <lightblue>.  You may add the line `export LIGHTBLUE=<lightblue>` to .bashrc, .bash.profile, .bash_profile, or whatever configuration file for your shell.

Then move to <lightblue>, create a sandbox environment there, and check the dependencies as follows.
```
$ cd <lightblue>
$ cabal sandbox init
$ cabal install --only-dependencies
```
If everything is ok, then build *lightblue* there.

```
$ cabal build
```
If the build is successful, then you may install *lightblue-0.1.6.0* in the sandbox.
```
$ cabal install
```
If succeeded, executable is found at `<lightblue>/.cabal-sandbox/bin/lightblue`.  You may want to add a path to `<lightblue>/.cabal-sandbox/bin`.

### Generating the Document
The HTML document is created by the following command in <lightblue>:
```
$ cabal --executable haddock
```
The generated document is found at: `<lightblue>/dist/doc/html/lightblue/index.html`

### How to run tests
Set the permission of two shell scripts `parse` and `tidy` to executable.
```
$ chmod 755 parse
$ chmod 755 tidy
```

To parse a Japanese sentence and get a text|HTML|TeX|XML representation, execute:
```
$ echo 太郎がパンを食べた。 | lightblue -o {text|HTML|TeX}
```
or
```
$ echo 太郎がパンを食べた。 | lightblue -o XML | ./tidy
```

With '-n|--nbest' option, *lightblue* will show the N-best parse results.

With '--time' option, *lightblue* will show the execution time for parsing.

*lightblue* can be used as a part-of-speech tagger when the `-postag` option is specified:
```
$ echo 太郎がパンを食べた。 | lightblue -t postag
```

The following command shows the list of lexical items prepared for pasing the given sentence:
```
$ echo 太郎がパンを食べた。| lightblue --t numeration
```

If you have a text file (one sentence per line) <corpusfile>, then you can feed it to *lightblue* by:
```
$ lightblue --corpus  <corpusfile>
```

To parse a JSeM file and execute inferences therein, then you can feed it to *lightblue* by:
```
$ lightblue --jsem  <jsemfile>
```

Check also:
```
$ lightblue --help
$ lightblue --version
$ lightblue --stat
```

## Contact ##

* Repo owner: [Daisuke Bekki](http://www.is.ocha.ac.jp/~bekki/)
* [Learn Markdown](https://bitbucket.org/tutorials/markdowndemo)