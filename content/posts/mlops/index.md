---
title: MLOps End to End Pipeline
description: Created end to end workflow that helps in managing different stages of machine learning projects. 
date: '2022-06-30'
draft: false
slug: '/blog_posts/mlops'
tags:
  - OpenCV
  - TensorFlow
  - Keras
  - Python
  - Html/CSS/JS
---

## Problem

Recently while updating with [Skela](https://github.com/Upstatement/skela-wp-theme) with webpack, I encountered a weird error where I wasn't able to run a simple script:

```shell:title=bin/composer
#!/bin/bash
docker-compose exec -w /var/www/html/wp-content/themes/skela wordpress composer "$@"
```

When trying to run this script via `./bin/composer install`, I got this error in my terminal:

```shell
ERROR: Setting workdir for exec is not supported in API < 1.35 (1.30)
```

The error was coming from the `-w` flag in the `docker-compose exec` command in the `composer` script.

## Solution

Turns The fix was to update the version in my `docker-compose.yml` file to from version `3.5` to `3.6`. It's strange because 3.5 isn't anywhere close to the API version `1.35` from the error message 🤷‍♀️

```yaml:title=docker-compose.yml
version: '3.6' # highlight-line
services:
  wordpress:
    build:
```
