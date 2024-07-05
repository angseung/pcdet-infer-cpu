#!/usr/bin/bash

if [ -n "$1" ]; then
        BUILD_NUMBER="-$1"
else
        BUILD_NUMBER=""
fi

REV_HASH=`git rev-list --tags --max-count=1 | head -c 7`
GIT_TAG=`git describe --tags --always $REV_HASH`
GIT_HASH=`git log -n 1 | head -n 1 | sed -e 's/^commit //' | head -c 7`
GIT_DIRTY=`git status -uno -s | grep -v 'go.mod' | grep -v 'go.sum'`
if [[ "$REV_HASH" != "$GIT_HASH" ]]; then
        GIT_TAG="latest-$GIT_HASH"
else
        GIT_TAG="$GIT_TAG"
fi

if [[ "$GIT_DIRTY" != "" ]]; then
        GIT_TAG="$GIT_TAG-dirty"
fi

BUILD_TIME=$(date +"%Y-%m-%dT%H:%M:%S%z")

# Create a version.h file with the git commit hash
echo "#ifndef VERSION_H" > ./include/version.h
echo "#define VERSION_H" >> ./include/version.h
echo "#define GIT_TAG_VERSION \"${GIT_TAG}${BUILD_NUMBER}\"" >> ./include/version.h
echo "#define BUILD_TIME \"$BUILD_TIME\"" >> ./include/version.h
echo "#endif // VERSION_H" >> ./include/version.h
