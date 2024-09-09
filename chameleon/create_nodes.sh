#!/bin/bash

# exit when any command fails
set -e

ACCESS=$3

source $ACCESS

function wait_lease {
	echo "Wait for $LEASE_NAME to be activated"
	timeout=$1
	lease_status=$(blazar lease-show --format value -c status "$LEASE_NAME")
	while [[ $lease_status != "ACTIVE"  ]]; do
		sleep 5
		lease_status=$(blazar lease-show --format value -c status "$LEASE_NAME")
		timeout=$((timeout - 5))
		if (( timeout < 0 )); then
			echo "Wait too long for Lease $LEASE_NAME"
			exit 1
		fi
	done

	echo "Lease $LEASE_NAME is ready for business"
}

NUM_NODES=$1

NODE_TYPE=compute_haswell
LEASE_NAME="baremetal-lease1"
SHARED_NETWORK_ID="$(openstack network show sharednet1 -f value -c id)"

st="$(TZ="UTC" date +'%Y-%m-%d %H:%M')"
if [[ $OSTYPE == "darwin"* ]]; then
	ed="$(TZ="UTC" date -v+1d +'%Y-%m-%d %H:%M')"
else
	ed="$(TZ="UTC" date +'%Y-%m-%d %H:%M' -d'+1 day')"
fi

blazar lease-create \
	--physical-reservation min=$NUM_NODES,max=$NUM_NODES,resource_properties="[\"=\", \"\$node_type\", \"$NODE_TYPE\"]" \
	--start-date "$st" \
	--end-date "$ed" \
	"$LEASE_NAME"

wait_lease 300

RSRV_ID="$(blazar lease-show $LEASE_NAME -f value -c reservations | grep "id" -m 1 | awk -F "\"" '{print $4}')"

IMG_NAME=$2

for (( i = 0; i < NUM_NODES; i++ )); do
	SERVER_NAME="node$i"
	openstack server create \
		--image $IMG_NAME \
		--flavor baremetal \
		--key-name "Mac" \
		--nic net-id=$SHARED_NETWORK_ID \
		--hint reservation=$RSRV_ID \
		$SERVER_NAME
done

