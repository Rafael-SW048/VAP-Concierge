#!/usr/bin/env bash

unset OS_STORAGE_URL
unset OS_AUTH_URL
unset OS_AUTH_TYPE
unset OS_IDENTITY_API_VERSION
unset OS_PROJECT_ID
unset OS_REGION
unset OS_TOKEN

export OS_AUTH_TYPE=v3applicationcredential
export OS_AUTH_URL=https://chi.edge.chameleoncloud.org:5000/v3
export OS_IDENTITY_API_VERSION=3
export OS_REGION_NAME="CHI@Edge"
export OS_INTERFACE=public
export OS_APPLICATION_CREDENTIAL_ID=85b2483b76ef41f688fe05e678c60e19
export OS_APPLICATION_CREDENTIAL_SECRET=2nNQwfYeQK9i-nOcZhsHzX71h2Q20RqUQS5cbIKfD8gRAArb_O0rDdsskAjcSTGbQVWHLIx16djjmyhtxdnzJg
