---
title: 1x1 Convolution
description: Why does someone use a 1x1 convolution? And why is it called 1x1? It sounds like a pixel-sized joke! When it first showed up, people probably thought it was a prank. But hold on, let's dive deep into this mysterious little guy and unravel the math magic behind it.
date: '2024-07-11'
draft: false
slug: '/blog_posts/OneXOne_Convolution'
tags:
  - CNN
  - TensorFlow
---

## Overview
Recently while I was reading about the Self Attention Model being that was used over images. I heard about this term, 1x1 convolutions. It was beng used in calculating the Key, Query and Value. I first find it funny but when I went deeper to understand this guy. I was impressed.
In crafted a three step process to understand this concept.

### Why we even need this?
Well 

> Lorem ipsum dolor sit amet, consectetuer adipiscing elit. Aliquam hendrerit mi posuere lectus. Vestibulum enim wisi, viverra nec, fringilla in, laoreet vitae, risus.

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
