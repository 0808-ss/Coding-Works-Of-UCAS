// Copyright (c) 2022 Huawei Technologies Co.,Ltd. All rights reserved.
//
// sysMaster is licensed under Mulan PSL v2.
// You can use this software according to the terms and conditions of the Mulan
// PSL v2.
// You may obtain a copy of Mulan PSL v2 at:
//         http://license.coscl.org.cn/MulanPSL2
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
// NON-INFRINGEMENT, MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
// See the Mulan PSL v2 for more details.

use crate::config::Config;
use mio::{unix::SourceFd, Events, Interest, Poll, Token};
use nix::sys::{
    signal::{kill, Signal},
    signalfd::{SfdFlags, SigSet, SignalFd},
    socket::{getsockopt, sockopt::PeerCredentials},
    stat,
    time::{TimeSpec, TimeValLike},
    timerfd::ClockId,
    timerfd::{Expiration, TimerFd, TimerFlags, TimerSetTimeFlags},
    wait::{waitid, Id, WaitPidFlag, WaitStatus},
};
use nix::unistd::{execv, Pid};
use std::os::unix::io::AsRawFd;
use std::{
    ffi::CString,
    fs::{self, File},
    io::{self, Read},
    os::unix::{net::UnixListener, prelude::FileTypeExt},
    path::{Path, PathBuf},
    process::Command,
    time::Duration,
};

#[cfg(not(test))]
pub const INIT_SOCK: &str = "/run/sysmaster/init.sock";
#[cfg(test)]
pub const INIT_SOCK: &str = "init.sock";

pub static SIGNALS: [Signal; 4] = [
    Signal::SIGINT,
    Signal::SIGTERM,
    Signal::SIGCHLD,
    Signal::SIGHUP,
];

const ALLFD_TOKEN: Token = Token(0);
const TIMERFD_TOKEN: Token = Token(1);
const SIGNALFD_TOKEN: Token = Token(2);
const SOCKETFD_TOKEN: Token = Token(3);

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InitState {
    Init,
    Running,
    Reexec,
}

pub struct Runtime {
    poll: Poll,
    timerfd: TimerFd,
    signalfd: SignalFd,
    socketfd: UnixListener,
    config: Config,
    state: InitState,
    // sysmaster pid
    pid: u32,
    // sysmaster status
    online: bool,
    deserialize: bool,
}

impl Runtime {
    pub fn new() -> std::io::Result<Self> {
        // parse arguments, --pid, Invisible to user
        let mut pid = 0u32;
        let mut args = std::env::args().skip(1);
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--pid" => {
                    if let Some(value) = args.next() {
                        if value.starts_with('-') {
                            panic!("Missing or invalid value for option.");
                        }
                        pid = match value.parse::<u32>() {
                            Ok(v) => v,
                            Err(e) => panic!("Invalid value: {:?}", e),
                        };
                    } else {
                        panic!("Missing value for option --pid.");
                    }
                }
                _ => {
                    log::debug!("Unknown items: {}, ignored!", arg);
                }
            }
        }

        // check socket
        if let Ok(metadata) = fs::metadata(INIT_SOCK) {
            if metadata.file_type().is_socket() {
                fs::remove_file(INIT_SOCK)?;
            }
        }
        let sock_path = PathBuf::from(INIT_SOCK);
        let sock_parent = sock_path.parent().unwrap();
        if !sock_parent.exists() {
            let old_mask = stat::umask(stat::Mode::from_bits_truncate(!0o755));
            fs::create_dir_all(sock_parent)?;
            let _ = stat::umask(old_mask);
        }
        if fs::metadata(INIT_SOCK).is_ok() {
            let _ = fs::remove_file(INIT_SOCK);
        }
        let socketfd = UnixListener::bind(INIT_SOCK)?;

        // add signal
        let mut mask = SigSet::empty();
        for sig in SIGNALS.iter() {
            mask.add(*sig);
        }
        mask.thread_set_mask()?;
        let signalfd = SignalFd::with_flags(&mask, SfdFlags::SFD_CLOEXEC | SfdFlags::SFD_NONBLOCK)?;

        // set timer
        let timerfd = TimerFd::new(
            ClockId::CLOCK_MONOTONIC,
            TimerFlags::TFD_NONBLOCK | TimerFlags::TFD_CLOEXEC,
        )?;
        timerfd.set(
            Expiration::OneShot(TimeSpec::from_duration(Duration::from_nanos(1))),
            TimerSetTimeFlags::empty(),
        )?;

        // parse config
        let config = Config::load(None)?;

        Ok(Self {
            poll: Poll::new()?,
            timerfd,
            signalfd,
            socketfd,
            config,
            state: InitState::Init,
            pid,
            online: false,
            deserialize: pid != 0,
        })
    }

    pub fn register(&mut self, token: Token) -> std::io::Result<()> {
        let signalfd = self.signalfd.as_raw_fd();
        let mut signal_source = SourceFd(&signalfd);
        let timerfd = self.timerfd.as_raw_fd();
        let mut time_source = SourceFd(&timerfd);
        let socketfd = self.socketfd.as_raw_fd();
        let mut unix_source = SourceFd(&socketfd);

        match token {
            SIGNALFD_TOKEN => self.poll.registry().register(
                &mut signal_source,
                SIGNALFD_TOKEN,
                Interest::READABLE,
            )?,
            TIMERFD_TOKEN => self.poll.registry().register(
                &mut time_source,
                TIMERFD_TOKEN,
                Interest::READABLE,
            )?,
            SOCKETFD_TOKEN => self.poll.registry().register(
                &mut unix_source,
                SOCKETFD_TOKEN,
                Interest::READABLE,
            )?,
            _ => {
                self.poll.registry().register(
                    &mut signal_source,
                    SIGNALFD_TOKEN,
                    Interest::READABLE,
                )?;
                self.poll.registry().register(
                    &mut time_source,
                    TIMERFD_TOKEN,
                    Interest::READABLE,
                )?;
                self.poll.registry().register(
                    &mut unix_source,
                    SOCKETFD_TOKEN,
                    Interest::READABLE,
                )?;
            }
        }

        Ok(())
    }

    pub fn deregister(&mut self, token: Token) -> std::io::Result<()> {
        let signalfd = self.signalfd.as_raw_fd();
        let mut signal_source = SourceFd(&signalfd);
        let timerfd = self.timerfd.as_raw_fd();
        let mut time_source = SourceFd(&timerfd);
        let socketfd = self.socketfd.as_raw_fd();
        let mut unix_source = SourceFd(&socketfd);

        match token {
            SIGNALFD_TOKEN => self.poll.registry().deregister(&mut signal_source)?,
            TIMERFD_TOKEN => self.poll.registry().deregister(&mut time_source)?,
            SOCKETFD_TOKEN => {
                self.poll.registry().deregister(&mut unix_source)?;
            }
            _ => {
                self.poll.registry().deregister(&mut signal_source)?;
                self.poll.registry().deregister(&mut time_source)?;
                self.poll.registry().deregister(&mut unix_source)?;
            }
        }

        if [SOCKETFD_TOKEN, ALLFD_TOKEN].contains(&token) && fs::metadata(INIT_SOCK).is_ok() {
            fs::remove_file(INIT_SOCK)?;
        };

        Ok(())
    }

    fn load_config(&mut self) -> std::io::Result<()> {
        self.config = match Config::load(None) {
            Ok(c) => c,
            Err(e) => {
                log::error!("Failed to load config, error: {:?}, ignored!", e);
                return Ok(());
            }
        };
        Ok(())
    }

    /* 带来的优点：
    传统的处理方式：轮询检查 // 阻塞调用 // 信号驱动
    本次的处理方式：非阻塞事件驱动 // 高效的事件循环（使用 mio 库） // 及时回收 // 错误处理（记录错误而不是打断系统）// 
    资源利用（不会让父进程在等待子进程退出时阻塞）// 减少信号使用（可以避免信号处理中的竞态条件和丢失信号的问题）
    */
    fn reap_zombies(&self) {
        // peek signal

        // 结合使用 WNOHANG 和 WNOWAIT 标志位，使得 waitid 调用不会阻塞父进程，即使没有子进程退出。
        // 父进程可以继续执行其他任务，同时周期性地检查是否有子进程终止，而不会在 waitid 调用上浪费任何时间。

        // 定义位置 : use nix::sys::wait::{waitid, Id, WaitPidFlag, WaitStatus};
        let flags = WaitPidFlag::WEXITED | WaitPidFlag::WNOHANG | WaitPidFlag::WNOWAIT;
        



        // 创建了一个循环
        loop {
            // 使用 waitid 函数来检查是否有子进程已经终止
            // linux的传统函数int waitid(idtype_t idtype, id_t id, siginfo_t *infop, int options);
            // waitid 函数默认是阻塞式的，它会一直阻塞调用线程，直到满足以下条件之一：
            // （1）出现了一个与 idtype 和 id 参数指定的子进程集合匹配的状态改变。 （2）状态改变与 options 参数指定的选项匹配。
            // 也可以在需要非阻塞行为时通过设置 WNOHANG 标志来避免阻塞。（这就与master的思路相似）
            let wait_status = match waitid(Id::All, flags) {
                Ok(status) => status,
                // 记录错误信息并继续循环
                Err(e) => {
                    log::warn!("Error when waitid for all, {}", e);
                    continue;
                }
            };
            // 这段代码的目的是检查 waitid 调用的结果，并根据子进程的状态创建一个包含有用信息的元组 Option。
            let si = match wait_status {
                WaitStatus::Exited(pid, code) => Some((pid, code, Signal::SIGCHLD)),
                WaitStatus::Signaled(pid, signal, _dc) => Some((pid, -1, signal)),
                _ => None, // ignore
            };

            // pop: recycle the zombie
            if let Some((pid, _, _)) = si {
                // 这句话会回收子进程的资源
                if let Err(e) = waitid(Id::Pid(pid), WaitPidFlag::WEXITED) {
                    log::error!("Error when reap the zombie({:?}), ignored: {:?}!", pid, e);
                } 
            }
        }
    }

    pub fn handle_signal(&mut self) -> std::io::Result<()> {
        let sig = match self.signalfd.read_signal()? {
            Some(s) => s,
            None => return Ok(()),
        };
        match Signal::try_from(sig.ssi_signo as i32)? {
            Signal::SIGHUP => self.reload()?,
            Signal::SIGINT => {
                log::debug!("Received SIGINT for pid({:?})", sig.ssi_pid);
                self.exit(1);
            }
            Signal::SIGKILL => {
                self.kill_sysmaster();
            }
            Signal::SIGTERM => self.state = InitState::Reexec,
            Signal::SIGCHLD => self.reap_zombies(),
            _ => {
                log::debug!(
                    "Received signo {:?} for pid({:?}), ignored!",
                    sig.ssi_signo,
                    sig.ssi_pid
                );
            }
        };
        Ok(())
    }

    // 处理定时器事件，用于定期检查 sysmaster-core 的状态，并在必要时重新启动它
    pub fn handle_timer(&mut self) -> std::io::Result<()> {
        if self.config.timecnt == 0 {
            log::error!(
                "Keepalive: we tried multiple times, and no longer start {:?}.",
                self.config.bin
            );
            self.deregister(TIMERFD_TOKEN)?;
            self.deregister(SOCKETFD_TOKEN)?;
            return Ok(());
        }

        if self.online {
            self.online = false;
        } else {
            self.start_bin();
            self.config.timecnt -= 1;
        }
        self.timerfd.set(
            Expiration::OneShot(TimeSpec::seconds(self.config.timewait as i64)),
            TimerSetTimeFlags::empty(),
        )?;
        Ok(())
    }


    // handle_socket 方法 
    // 配合master_core的联网模块
    pub fn handle_socket(&mut self) -> std::io::Result<()> {
        let (stream, _) = match self.socketfd.accept() {
            Ok((connection, address)) => (connection, address),
            Err(e) if e.kind() == io::ErrorKind::WouldBlock => {
                // If we get a `WouldBlock` error we know our
                // listener has no more incoming connections queued,
                // so we can return to polling and wait for some
                // more.
                return Ok(());
            }
            Err(e) => {
                // If it was any other kind of error, something went
                // wrong and we terminate with an error.
                log::error!("Error accepting connection: {}!", e);
                return Err(e);
            }
        };
        
        // 检查连接的 PID 是否是 sysmaster-core 进程，如果是，则更新 online 和 pid
        let credentials = getsockopt(stream.as_raw_fd(), PeerCredentials)?;
        let pid = credentials.pid() as u32;
        if self.pid_is_running(pid) {
            // If the incoming PID is not the monitored sysmaster,
            // do not refresh the status.
            self.online = true;
            self.pid = pid;
        }
        Ok(())
    }

    // 构建 /proc/[pid]/comm 文件的路径，该文件包含进程的名称
    fn pid_is_running(&self, pid: u32) -> bool {
        let path = format!("/proc/{}/comm", pid);
        let file = Path::new(&path);
        if file.exists() {
            let mut content = String::new();
            let file = File::open(file);
            match file.map(|mut f| f.read_to_string(&mut content)) {
                Ok(_) => (),
                Err(_) => return false,
            };
            if content.starts_with("sysmaster") {
                return true;
            }
        }

        false
    }

    // 这个模块负责启动配置中指定的二进制文件
    fn start_bin(&mut self) {
        // check sysmaster status, if it is running then sigterm it
        if self.pid != 0 && (self.deserialize || self.kill_sysmaster()) {
            return;
        }

        // else start the binary
        let mut parts = self.config.bin.split_whitespace();
        let command = match parts.next() {
            Some(c) => c,
            None => {
                log::error!("Wrong command: {:?}!", self.config.bin);
                return;
            }
        };
        let args: Vec<&str> = parts.collect();
        if !Path::new(command).exists() {
            log::error!("Command {:?} does not exist!", command);
        }
        
        // 这段是在尝试创建新的子进程
        let child_process = match Command::new(command).args(&args).spawn() {
            Ok(child) => child,
            Err(_) => {
                log::error!("Failed to spawn process: {:?}.", command);
                return;
            }
        };

        self.pid = child_process.id();
        log::info!("Success to start {}({}))!", self.config.bin, self.pid);
    }

    // 这是master-core的监听循环
    // 所谓的事件源：如信号文件描述符（signalfd）、定时器文件描述符（timerfd）和 Unix 套接字监听器（UnixListener）
    // 具体的注册方式：使用 register 方法将这些事件源注册到 Poll 实例。每个事件源都与一个 Token 关联，用于在事件发生时标识它。
    // 具体的注册函数没有详细标注

    pub fn runloop(&mut self) -> std::io::Result<()> {
        // 注册事件源
        self.register(ALLFD_TOKEN)?;
        let mut events = Events::with_capacity(16);

        // event loop.
        loop {
            if !self.is_running() {
                self.deregister(ALLFD_TOKEN)?;
                break;
            }
            
            // 使用poll方法监测事件
            self.poll.poll(&mut events, None)?;

            // 根据事件类型调用相应的处理函数
            // Process each event.
            for event in events.iter() {
                match event.token() {
                    /*
                    SIGNALFD_TOKEN: 处理信号。
                    TIMERFD_TOKEN: 处理计时器事件。
                    SOCKETFD_TOKEN: 处理套接字事件。
                    */
                    SIGNALFD_TOKEN => self.handle_signal()?,
                    TIMERFD_TOKEN => self.handle_timer()?,
                    SOCKETFD_TOKEN => self.handle_socket()?,
                    _ => unreachable!(),
                }
            }

            #[cfg(test)]
            self.set_state(InitState::Init);
        }

        Ok(())
    }

    pub fn is_running(&self) -> bool {
        self.state == InitState::Running
    }

    pub fn set_state(&mut self, state: InitState) {
        self.state = state;
    }

    fn reload(&mut self) -> std::io::Result<()> {
        log::info!("Reloading init configuration!");
        self.load_config()?;
        Ok(())
    }

    pub fn is_reexec(&self) -> bool {
        self.state == InitState::Reexec
    }

    pub fn reexec(&mut self) {
        // Get the current executable path
        let exe = match std::env::current_exe().unwrap().file_name() {
            Some(v) => v.to_string_lossy().to_string(),
            None => "".to_string(),
        };

        for arg0 in [&exe, "/init", "/sbin/init"] {
            let argv = vec![arg0.to_string(), "--pid".to_string(), self.pid.to_string()];

            // Convert the argument and argument vector to CStrings
            let cstr_arg0 = CString::new(arg0).unwrap();
            let cstr_argv = argv
                .iter()
                .map(|arg| CString::new(arg.as_str()).unwrap())
                .collect::<Vec<_>>();

            log::info!("Reexecuting init: {:?}", argv);

            // Execute the new process
            if let Err(e) = execv(&cstr_arg0, &cstr_argv) {
                log::error!("Execv {:?} {:?} failed: {:?}", arg0, argv, e);
            }
        }
    }

    fn kill_sysmaster(&mut self) -> bool {
        if self.pid_is_running(self.pid) {
            let target_pid = Pid::from_raw(self.pid.try_into().unwrap());

            match kill(target_pid, Signal::SIGTERM) {
                Ok(_) => {
                    log::info!(
                        "Timeout, send SIGTERM to {} ({})!",
                        self.config.bin,
                        self.pid
                    );
                    return true;
                }
                Err(err) => log::error!(
                    "Timeout, failed to send SIGTERM to {} ({}), {}, ignore!",
                    self.config.bin,
                    self.pid,
                    err
                ),
            }
        }
        false
    }

    fn exit(&self, i: i32) {
        std::process::exit(i);
    }
}

impl Drop for Runtime {
    fn drop(&mut self) {
        if fs::metadata(INIT_SOCK).is_ok() {
            let _ = fs::remove_file(INIT_SOCK);
        }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_runtime() -> std::io::Result<()> {
        use crate::runtime::*;
        let mut rt = Runtime::new()?;
        rt.set_state(InitState::Running);
        rt.config.timewait = 0;
        rt.runloop()?;
        assert_ne!(rt.timerfd.as_raw_fd(), 0);
        assert_ne!(rt.signalfd.as_raw_fd(), 0);
        assert_ne!(rt.socketfd.as_raw_fd(), 0);
        Ok(())
    }
}
