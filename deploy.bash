#!/usr/bin/env bash

image_name="ds9"
version=$(head -n 1 "VERSION")
push=""
latest=""
test=""

while [[ $# -gt 0 ]]; do
  case $1 in
    -i|--image-name)
      image_name=$2
      shift
      shift
      ;;
    -v|--version)
      version=$2
      shift
      shift
      ;;
    -p|--push)
      push="yes"
      shift
      ;;
    -l|--latest)
      latest="yes"
      shift
      ;;
    -t|--test)
      test="yes"
      shift
      ;;
  esac
done

image_name="ghcr.io/klebert-engineering/$image_name"
docker build -t "$image_name:$version" .

if [[ -n "$latest" ]]; then
  echo "Tagging latest."
  docker tag "$image_name:$version" "$image_name:latest"
fi

if [[ -n "$push" ]]; then
  echo "Pushing."
  docker push "$image_name:$version"
  if [[ -n "$latest" ]]; then
    docker push "$image_name:latest"
  fi
fi

if [[ -n "$test" ]]; then
  echo "Running test."
  docker run --rm -it \
    -p 8091:8091 \
    "$image_name:$version" \
    "//ds9/serve.bash"
fi
py