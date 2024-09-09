----
-- User configuration file for lsyncd.
--
-- Simple example for default rsync, but executing moves through on the target.
--
settings {
	logfile = "/var/log/lsyncd/lsyncd.log",
	statusFile = "/var/log/lsyncd/lsyncd.status",
	statusInterval = 1,
	nodaemon   = false
}

sync {
	default.rsyncssh,
	source = "/home/cc/vap-concierge",
	host = "royhuang@107.199.157.238",
	targetdir = "vap-concierge",
	delay = 5,
	rsync = {
		rsh="/usr/bin/ssh -p 10022 -i /home/cc/.ssh/id_rsa -o StrictHostKeyChecking=no -l royhuang",
		_extra={"--rsync-path=/usr/bin/rsync"}
	}
}
