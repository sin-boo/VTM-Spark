// run.exe - windowless launcher (compiled /target:winexe by build-run.ps1).
// It never opens a console on its own: leftover cleanup runs in process, a
// stale-UI rebuild and pythonw -m backend run hidden, and it exits once the
// desk's splash window is up. When a start genuinely fails, a native dialog
// shows the reason and offers the repair console (start-menu.ps1 -Action run);
// the console only appears if the user says yes.
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Management;
using System.Net;
using System.Net.NetworkInformation;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Security.Cryptography.X509Certificates;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading;

[assembly: AssemblyTitle("VTM Spark")]
[assembly: AssemblyDescription("VTM Spark launcher")]
[assembly: AssemblyCompany("VTM Studio")]
[assembly: AssemblyProduct("VTM Spark")]
[assembly: AssemblyCopyright("Copyright (c) VTM Studio")]
[assembly: AssemblyVersion("1.0.0.0")]
[assembly: AssemblyFileVersion("1.0.0.0")]

internal static class Program
{
    [DllImport("user32.dll")] private static extern bool SetForegroundWindow(IntPtr hWnd);
    [DllImport("user32.dll")] private static extern bool ShowWindow(IntPtr hWnd, int nCmdShow);
    [DllImport("user32.dll")] private static extern bool IsIconic(IntPtr hWnd);
    [DllImport("user32.dll")] private static extern bool AllowSetForegroundWindow(int pid);
    private const int ASFW_ANY = -1;
    [DllImport("user32.dll", CharSet = CharSet.Unicode)]
    private static extern int MessageBoxW(IntPtr hWnd, string text, string caption, uint type);

    private delegate bool EnumWindowsProc(IntPtr hWnd, IntPtr lParam);
    [DllImport("user32.dll")] private static extern bool EnumWindows(EnumWindowsProc callback, IntPtr lParam);
    [DllImport("user32.dll")] private static extern bool IsWindowVisible(IntPtr hWnd);
    [DllImport("user32.dll")] private static extern IntPtr GetWindow(IntPtr hWnd, uint cmd);
    [DllImport("user32.dll")] private static extern uint GetWindowThreadProcessId(IntPtr hWnd, out uint pid);
    [DllImport("user32.dll", CharSet = CharSet.Unicode)]
    private static extern int GetWindowText(IntPtr hWnd, StringBuilder text, int max);
    private const uint GW_OWNER = 4;

    [DllImport("iphlpapi.dll")]
    private static extern uint GetExtendedTcpTable(IntPtr table, ref int size, bool order, int af, int tableClass, uint reserved);
    private const int TCP_TABLE_OWNER_PID_LISTENER = 3;

    private const string DeskTitle = "VTM Spark";
    private const string DeskMutex = "Local\\VTMNobleSingleInstance";
    // Held by run.exe while it launches, before the desk takes DeskMutex.
    private const string LaunchMutex = "Local\\VTMNobleLauncher";
    private const int DeskPort = 8765;
    // Last port kill-orphans.ps1's full sweep clears (8765-8828).
    private const int LastPort = 8828;
    // backend/__main__.py exits with this when WebView2 is not installed.
    private const int WebView2MissingCode = 3;
    private const string WebView2RuntimeKey = "{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}";
    private const string WebView2Bootstrapper = "https://go.microsoft.com/fwlink/p/?LinkId=2124703";

    private const uint MB_OK = 0x00000000;
    private const uint MB_YESNO = 0x00000004;
    private const uint MB_ICONWARNING = 0x00000030;
    private const uint MB_ICONINFORMATION = 0x00000040;
    private const uint MB_SETFOREGROUND = 0x00010000;
    private const uint MB_TOPMOST = 0x00040000;
    private const int IDYES = 6;

    private static string Root = "";

    [STAThread]
    private static int Main()
    {
        string here = Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location);
        if (string.IsNullOrEmpty(here))
        {
            here = Environment.CurrentDirectory;
        }
        Root = here;
        Directory.SetCurrentDirectory(here);

        bool first;
        using (Mutex launching = new Mutex(true, LaunchMutex, out first))
        {
            if (!first)
            {
                // A click moments ago is still starting the desk.
                FocusDesk(TimeSpan.FromSeconds(3));
                return 0;
            }
            return Launch(here);
        }
    }

    private static int Launch(string here)
    {
        if (DeskRunning())
        {
            // Second click: bring the open (or still booting) desk forward.
            if (FocusDesk(TimeSpan.FromSeconds(3)))
            {
                return 0;
            }
            // No window: usually a desk that was just closed and is still
            // freeing GPU memory. Let it finish before calling it a hidden copy.
            if (!WaitDeskExit(TimeSpan.FromSeconds(10)))
            {
                if (FocusDeskOnce())
                {
                    return 0;
                }
                Log("single-instance lock held but no desk window found");
                if (!Ask("VTM Spark looks like it is already running, but its window cannot be found.\n\n"
                    + "Clean up leftover processes and start it again?"))
                {
                    return 0;
                }
                KillOrphans(false);
            }
        }

        string pyw = DeskPython(here);
        if (pyw == null)
        {
            Log("no usable .venv-build python");
            return Fail("No working Python environment (.venv-build).\nRun install.bat first.");
        }

        string failure;
        try
        {
            failure = Boot(pyw);
        }
        catch (Exception ex)
        {
            Log("start threw: " + ex.Message);
            failure = "Could not start the desk:\n" + ex.Message;
        }
        return failure == null ? 0 : Fail(failure);
    }

    // Everything between the click and the desk's splash. Null = started.
    private static string Boot(string pyw)
    {
        if (UiStale(Root) && !BuildUi())
        {
            return "The desk UI is not built and the rebuild failed.\nRun install.bat first.";
        }

        // Same sweep start-menu.ps1 does before a start: a leftover backend or
        // ui/_mock_boot.py on port 8765 would otherwise refuse the desk. It
        // needs a WMI query, so it only runs when something could be left over.
        if (MaybeLeftovers())
        {
            KillOrphans(true);
        }
        // install.bat installs WebView2; this covers a skipped or failed step.
        if (!WebView2Installed())
        {
            InstallWebView2();
        }
        return StartDesk(pyw);
    }

    // Cheap check for anything KillOrphans(true) could stop.
    private static bool MaybeLeftovers()
    {
        foreach (string name in new[] { "python", "pythonw", "VTMNoble", "RealStream" })
        {
            Process[] found = Process.GetProcessesByName(name);
            bool any = found.Length > 0;
            foreach (Process proc in found)
            {
                proc.Dispose();
            }
            if (any)
            {
                return true;
            }
        }
        try
        {
            foreach (IPEndPoint ep in IPGlobalProperties.GetIPGlobalProperties().GetActiveTcpListeners())
            {
                if (ep.Port == DeskPort)
                {
                    return true;
                }
            }
        }
        catch (Exception)
        {
            return true;
        }
        return false;
    }

    // Null = the desk's window is up; otherwise why it is not.
    private static string StartDesk(string pyw)
    {
        Process p;
        try
        {
            ProcessStartInfo psi = new ProcessStartInfo
            {
                FileName = pyw,
                Arguments = "-m backend --ui webview",
                WorkingDirectory = Root,
                UseShellExecute = false,
                CreateNoWindow = true,
            };
            psi.EnvironmentVariables["PYTHONPATH"] = Root;
            p = Process.Start(psi);
        }
        catch (Exception ex)
        {
            Log("could not start pythonw: " + ex.Message);
            return "Could not start the desk:\n" + ex.Message;
        }
        if (p == null)
        {
            return "Could not start the desk.";
        }
        // The click made this process the foreground one; pass that on so the
        // desk (a grandchild, behind the venv launcher) may come to the front.
        AllowSetForegroundWindow(ASFW_ANY);

        // Stay until the desk's splash is up so a failed start still gets
        // reported. The venv's pythonw.exe is only a launcher: the real
        // interpreter is its child and owns the window, so look for the window
        // itself; the launcher's exit code is still the desk's.
        DateTime deadline = DateTime.UtcNow.AddSeconds(60);
        while (DateTime.UtcNow < deadline)
        {
            if (p.WaitForExit(150))
            {
                if (p.ExitCode == 0)
                {
                    return null;
                }
                Log("desk exited with code " + p.ExitCode + " before its window opened");
                if (p.ExitCode == WebView2MissingCode)
                {
                    return "The Microsoft Edge WebView2 runtime could not be installed automatically.\n\n"
                        + "Check your internet connection and open VTM Spark again.";
                }
                return "The desk exited before its window opened (code " + p.ExitCode + ").";
            }
            if (FindDeskWindow(true) != IntPtr.Zero)
            {
                return null;
            }
        }
        Log("desk window did not appear within 60 s");
        return null;
    }

    // Same registry keys as pywebview and start-menu.ps1 Test-WebView2.
    private static bool WebView2Installed()
    {
        string client = @"Microsoft\EdgeUpdate\Clients\" + WebView2RuntimeKey;
        string[] keys =
        {
            @"HKEY_LOCAL_MACHINE\SOFTWARE\WOW6432Node\" + client,
            @"HKEY_LOCAL_MACHINE\SOFTWARE\" + client,
            @"HKEY_CURRENT_USER\Software\" + client,
        };
        foreach (string key in keys)
        {
            string pv = Microsoft.Win32.Registry.GetValue(key, "pv", null) as string;
            if (!string.IsNullOrEmpty(pv) && pv != "0.0.0.0")
            {
                return true;
            }
        }
        return false;
    }

    // Microsoft's Evergreen bootstrapper, kept in .tools\webview2 and reused:
    // silent per-user first (no UAC), then system-wide once, with its own UI,
    // if that did not take.
    private static void InstallWebView2()
    {
        Log("WebView2 runtime missing - installing");
        string exe = Path.Combine(Root, ".tools", "webview2", "MicrosoftEdgeWebview2Setup.exe");
        try
        {
            if (File.Exists(exe) && !SignedByMicrosoft(exe))
            {
                Log("discarding unsigned " + exe);
                File.Delete(exe);
            }
            if (!File.Exists(exe))
            {
                Directory.CreateDirectory(Path.GetDirectoryName(exe));
                ServicePointManager.SecurityProtocol = (SecurityProtocolType)3072; // TLS 1.2
                using (WebClient web = new WebClient())
                {
                    web.DownloadFile(WebView2Bootstrapper, exe);
                }
            }
            if (!SignedByMicrosoft(exe))
            {
                Log("downloaded WebView2 setup is not signed by Microsoft - deleted");
                File.Delete(exe);
                return;
            }
            RunWebView2Installer(exe, false);
            if (!WebView2Installed())
            {
                Inform("VTM Spark needs Microsoft WebView2 to show its window, and the quick install did not work.\n\n"
                    + "Windows will now ask for administrator permission to install it for all users. "
                    + "The prompt will say \"Microsoft Edge Update Setup\" - that is this step.\n\n"
                    + "Click Yes to continue.");
                RunWebView2Installer(exe, true);
            }
            Log(WebView2Installed() ? "WebView2 runtime installed" : "WebView2 runtime still missing");
        }
        catch (Exception ex)
        {
            Log("WebView2 install failed: " + ex.Message);
        }
    }

    // Authenticode signer check; also rejects a truncated download.
    private static bool SignedByMicrosoft(string file)
    {
        try
        {
            return X509Certificate.CreateFromSignedFile(file).Subject.Contains("O=Microsoft Corporation");
        }
        catch (Exception)
        {
            return false;
        }
    }

    private static void RunWebView2Installer(string exe, bool elevated)
    {
        ProcessStartInfo psi = new ProcessStartInfo
        {
            FileName = exe,
            Arguments = elevated ? "/install" : "/silent /install",
            UseShellExecute = elevated,
        };
        if (elevated)
        {
            psi.Verb = "runas";
        }
        using (Process p = Process.Start(psi))
        {
            if (p != null)
            {
                p.WaitForExit((int)TimeSpan.FromMinutes(10).TotalMilliseconds);
            }
        }
    }

    // Rebuild ui/dist hidden: portable Node runs npm-cli.js itself; cmd.exe
    // only when there is no .tools\node. True = ui/dist is usable (fresh, or
    // the last build when the rebuild could not run).
    private static bool BuildUi()
    {
        string ui = Path.Combine(Root, "ui");
        string index = Path.Combine(ui, "dist", "index.html");
        string node = Path.Combine(Root, ".tools", "node", "node.exe");
        string npmCli = Path.Combine(Root, ".tools", "node", "node_modules", "npm", "bin", "npm-cli.js");
        Log("ui source newer than ui\\dist - rebuilding hidden");
        TimeSpan limit = TimeSpan.FromMinutes(10);
        int code = File.Exists(node) && File.Exists(npmCli)
            ? RunHidden(node, "\"" + npmCli + "\" run build", ui, limit)
            : RunHidden("cmd.exe", "/c npm run build", ui, limit);
        if (code == 0)
        {
            Log("ui rebuilt");
            return true;
        }
        Log("ui rebuild failed (exit " + code + ")" + (File.Exists(index) ? "; launching last ui\\dist" : ""));
        return File.Exists(index);
    }

    // kill-orphans.ps1 -Quiet [-Fast], in process; start-menu.ps1 and build.ps1
    // still run the script, so keep the two in step.
    private static void KillOrphans(bool fast)
    {
        try
        {
            SweepOrphans(fast);
        }
        catch (Exception ex)
        {
            Log("orphan sweep failed: " + ex.Message);
        }
    }

    private sealed class Proc
    {
        public int Pid;
        public int Parent;
        public string Name;
        public string Cmd;
        public DateTime Created;
    }

    private const RegexOptions Ci = RegexOptions.IgnoreCase;
    private static readonly Regex PyExe = new Regex(@"^(python|pythonw)\.exe$", Ci);
    private static readonly Regex MBackend = new Regex(@"-m\s+backend(\s|$)", Ci);
    private static readonly Regex OurDir = new Regex(
        @"[\\/](vtm[ _-]?(noble|studio|spark)|real_stream|VTMNoble|VTMStudio|VTMSpark|RealStream)[\\/]", Ci);
    private static readonly Regex OurDist = new Regex(@"[\\/]dist[\\/](VTMNoble|RealStream)[\\/]", Ci);
    private static readonly Regex OurEnv = new Regex("VTM_NOBLE|REAL_STREAM", Ci);
    private static readonly Regex OurRuntime = new Regex(@"[\\/]dist[\\/]VTMNoble[\\/]runtime[\\/].*python", Ci);
    private static readonly Regex MockBoot = new Regex(@"_mock_boot\.py", Ci);

    private static void SweepOrphans(bool fast)
    {
        List<Proc> procs = Snapshot();
        foreach (string name in new[] { "VTMNoble", "RealStream" })
        {
            foreach (int pid in PidsNamed(name))
            {
                KillTree(pid, procs);
            }
            // taskkill /F /IM <name>.exe
            foreach (int pid in PidsNamed(name))
            {
                KillPid(pid);
            }
        }

        // Full: ...\python.exe -m backend from one of our trees, or the packaged
        // runtime. Fast: any python -m backend or ui/_mock_boot.py.
        foreach (Proc p in procs)
        {
            if (p.Name == null || !PyExe.IsMatch(p.Name) || string.IsNullOrEmpty(p.Cmd))
            {
                continue;
            }
            bool ours = fast
                ? MBackend.IsMatch(p.Cmd) || MockBoot.IsMatch(p.Cmd)
                : (MBackend.IsMatch(p.Cmd)
                    && (OurDir.IsMatch(p.Cmd) || OurDist.IsMatch(p.Cmd) || OurEnv.IsMatch(p.Cmd)))
                    || OurRuntime.IsMatch(p.Cmd);
            if (ours)
            {
                KillTree(p.Pid, procs);
            }
        }

        // Fast: python on 8765 (Vite's mock API has no "-m backend").
        // Full: python/desk on 8765-8828 (stale Track Lab on 8780 and the like).
        Regex owner = new Regex(fast ? "^(python|pythonw)" : "^(python|pythonw|VTMNoble|RealStream)", Ci);
        foreach (int pid in Listeners(DeskPort, fast ? DeskPort : LastPort))
        {
            if (pid <= 4)
            {
                continue;
            }
            string name = NameOf(pid);
            if (name != null && owner.IsMatch(name))
            {
                KillTree(pid, procs);
            }
        }
        if (!fast)
        {
            Thread.Sleep(500);
        }
    }

    // Win32_Process: name, command line and parent of every process.
    private static List<Proc> Snapshot()
    {
        List<Proc> procs = new List<Proc>();
        try
        {
            using (ManagementObjectSearcher q = new ManagementObjectSearcher(
                "SELECT ProcessId, ParentProcessId, Name, CommandLine, CreationDate FROM Win32_Process"))
            using (ManagementObjectCollection rows = q.Get())
            {
                foreach (ManagementBaseObject row in rows)
                {
                    using (row)
                    {
                        Proc p = new Proc
                        {
                            Pid = Convert.ToInt32(row["ProcessId"]),
                            Parent = Convert.ToInt32(row["ParentProcessId"]),
                            Name = row["Name"] as string,
                            Cmd = row["CommandLine"] as string,
                            Created = DateTime.MinValue,
                        };
                        string created = row["CreationDate"] as string;
                        if (!string.IsNullOrEmpty(created))
                        {
                            try { p.Created = ManagementDateTimeConverter.ToDateTime(created); } catch (Exception) { }
                        }
                        procs.Add(p);
                    }
                }
            }
        }
        catch (Exception ex)
        {
            Log("process list unavailable: " + ex.Message);
        }
        return procs;
    }

    private static List<int> PidsNamed(string name)
    {
        List<int> pids = new List<int>();
        foreach (Process proc in Process.GetProcessesByName(name))
        {
            pids.Add(proc.Id);
            proc.Dispose();
        }
        return pids;
    }

    private static string NameOf(int pid)
    {
        try
        {
            using (Process proc = Process.GetProcessById(pid))
            {
                return proc.ProcessName;
            }
        }
        catch (Exception)
        {
            return null;
        }
    }

    // taskkill /F /T /PID: the process, then everything it started. A child
    // older than its parent only inherited a reused pid and is left alone.
    private static void KillTree(int root, List<Proc> procs)
    {
        Dictionary<int, Proc> byPid = new Dictionary<int, Proc>();
        foreach (Proc p in procs)
        {
            byPid[p.Pid] = p;
        }
        List<int> doomed = new List<int> { root };
        for (int i = 0; i < doomed.Count; i++)
        {
            Proc parent;
            byPid.TryGetValue(doomed[i], out parent);
            foreach (Proc p in procs)
            {
                if (p.Parent != doomed[i] || doomed.Contains(p.Pid))
                {
                    continue;
                }
                if (parent != null && parent.Created != DateTime.MinValue
                    && p.Created != DateTime.MinValue && p.Created < parent.Created)
                {
                    continue;
                }
                doomed.Add(p.Pid);
            }
        }
        foreach (int pid in doomed)
        {
            KillPid(pid);
        }
    }

    private static void KillPid(int pid)
    {
        if (pid <= 4 || pid == Process.GetCurrentProcess().Id)
        {
            return;
        }
        try
        {
            using (Process proc = Process.GetProcessById(pid))
            {
                proc.Kill();
            }
        }
        catch (Exception) { }
    }

    // Owning pids of TCP listeners on ports lo..hi, IPv4 and IPv6
    // (Get-NetTCPConnection -State Listen).
    private static List<int> Listeners(int lo, int hi)
    {
        List<int> pids = new List<int>();
        // AF_INET rows: 24 bytes, port at 8, pid at 20; AF_INET6: 56, 20, 52.
        int[][] layouts = { new[] { 2, 24, 8, 20 }, new[] { 23, 56, 20, 52 } };
        foreach (int[] l in layouts)
        {
            for (int attempt = 0; attempt < 3; attempt++)
            {
                int size = 0;
                GetExtendedTcpTable(IntPtr.Zero, ref size, false, l[0], TCP_TABLE_OWNER_PID_LISTENER, 0);
                if (size <= 0)
                {
                    break;
                }
                IntPtr buf = Marshal.AllocHGlobal(size);
                try
                {
                    uint err = GetExtendedTcpTable(buf, ref size, false, l[0], TCP_TABLE_OWNER_PID_LISTENER, 0);
                    if (err == 122) // ERROR_INSUFFICIENT_BUFFER: the table grew
                    {
                        continue;
                    }
                    if (err == 0)
                    {
                        int count = Marshal.ReadInt32(buf);
                        for (int i = 0; i < count; i++)
                        {
                            IntPtr row = new IntPtr(buf.ToInt64() + 4 + (long)i * l[1]);
                            int raw = Marshal.ReadInt32(row, l[2]);
                            int port = ((raw & 0xFF) << 8) | ((raw >> 8) & 0xFF);
                            int pid = Marshal.ReadInt32(row, l[3]);
                            if (port >= lo && port <= hi && !pids.Contains(pid))
                            {
                                pids.Add(pid);
                            }
                        }
                    }
                    break;
                }
                finally
                {
                    Marshal.FreeHGlobal(buf);
                }
            }
        }
        return pids;
    }

    // Exit code of a hidden child; -1 = could not start, -2 = timed out (killed).
    private static int RunHidden(string file, string args, string cwd, TimeSpan timeout)
    {
        try
        {
            ProcessStartInfo psi = new ProcessStartInfo
            {
                FileName = file,
                Arguments = args,
                WorkingDirectory = cwd,
                UseShellExecute = false,
                CreateNoWindow = true,
            };
            psi.EnvironmentVariables["PYTHONPATH"] = Root;
            // Portable Node from install.bat (.tools\node) wins over a system one.
            string node = Path.Combine(Root, ".tools", "node");
            if (File.Exists(Path.Combine(node, "node.exe")) || File.Exists(Path.Combine(node, "npm.cmd")))
            {
                psi.EnvironmentVariables["PATH"] = node + ";" + psi.EnvironmentVariables["PATH"];
                psi.EnvironmentVariables["npm_config_cache"] = Path.Combine(Root, ".tools", "npm-cache");
            }
            using (Process p = Process.Start(psi))
            {
                if (p == null)
                {
                    return -1;
                }
                if (p.WaitForExit((int)timeout.TotalMilliseconds))
                {
                    return p.ExitCode;
                }
                try { p.Kill(); } catch (Exception) { }
                return -2;
            }
        }
        catch (Exception ex)
        {
            Log("could not run " + file + ": " + ex.Message);
            return -1;
        }
    }

    // Report a failed start in a dialog. Yes = open the repair console.
    private static int Fail(string reason)
    {
        StringBuilder sb = new StringBuilder(reason);
        string tail = LogTail(8);
        if (tail.Length > 0)
        {
            sb.Append("\n\nLast log lines:\n").Append(tail);
        }
        sb.Append("\n\nOpen the repair console?");
        if (Ask(sb.ToString()))
        {
            return RunMenu(Root);
        }
        return 1;
    }

    private static bool Ask(string text)
    {
        return MessageBoxW(IntPtr.Zero, text, DeskTitle,
            MB_YESNO | MB_ICONWARNING | MB_SETFOREGROUND | MB_TOPMOST) == IDYES;
    }

    // Heads-up before a Windows admin prompt, so it is not a surprise.
    private static void Inform(string text)
    {
        MessageBoxW(IntPtr.Zero, text, DeskTitle,
            MB_OK | MB_ICONINFORMATION | MB_SETFOREGROUND | MB_TOPMOST);
    }

    private static string LogPath()
    {
        return Path.Combine(Root, "models", "vtm_noble.log");
    }

    private static void Log(string msg)
    {
        try
        {
            string path = LogPath();
            Directory.CreateDirectory(Path.GetDirectoryName(path));
            File.AppendAllText(path, "[" + DateTime.Now.ToString("HH:mm:ss") + "] run.exe: " + msg + "\n");
        }
        catch (Exception) { }
    }

    private static string LogTail(int count)
    {
        try
        {
            string path = LogPath();
            if (!File.Exists(path))
            {
                return "";
            }
            string[] lines = File.ReadAllLines(path);
            List<string> keep = new List<string>();
            for (int i = Math.Max(0, lines.Length - count); i < lines.Length; i++)
            {
                string line = lines[i].TrimEnd();
                if (line.Length > 110)
                {
                    line = line.Substring(0, 107) + "...";
                }
                keep.Add(line);
            }
            return string.Join("\n", keep.ToArray());
        }
        catch (Exception)
        {
            return "";
        }
    }

    private static bool DeskRunning()
    {
        Mutex m;
        if (!Mutex.TryOpenExisting(DeskMutex, out m))
        {
            return false;
        }
        m.Dispose();
        return true;
    }

    // True once the desk's single-instance lock is gone (it finished exiting).
    private static bool WaitDeskExit(TimeSpan patience)
    {
        DateTime deadline = DateTime.UtcNow + patience;
        while (DeskRunning())
        {
            if (DateTime.UtcNow >= deadline)
            {
                return false;
            }
            Thread.Sleep(250);
        }
        return true;
    }

    private static bool FocusDesk(TimeSpan patience)
    {
        DateTime deadline = DateTime.UtcNow + patience;
        while (true)
        {
            if (FocusDeskOnce())
            {
                return true;
            }
            if (DateTime.UtcNow >= deadline)
            {
                return false;
            }
            Thread.Sleep(250);
        }
    }

    // The desk (or its splash, mid-launch) comes forward.
    private static bool FocusDeskOnce()
    {
        IntPtr hwnd = FindDeskWindow(false);
        if (hwnd == IntPtr.Zero)
        {
            return false;
        }
        if (IsIconic(hwnd))
        {
            ShowWindow(hwnd, 9); // SW_RESTORE
        }
        SetForegroundWindow(hwnd);
        return true;
    }

    // A visible top-level "VTM Spark" window from another process. With
    // pythonOnly, only the desk's own windows count.
    private static IntPtr FindDeskWindow(bool pythonOnly)
    {
        uint self = (uint)Process.GetCurrentProcess().Id;
        IntPtr found = IntPtr.Zero;
        StringBuilder title = new StringBuilder(64);
        EnumWindows(delegate (IntPtr hwnd, IntPtr unused)
        {
            if (!IsWindowVisible(hwnd) || GetWindow(hwnd, GW_OWNER) != IntPtr.Zero)
            {
                return true;
            }
            title.Length = 0;
            GetWindowText(hwnd, title, title.Capacity);
            if (title.ToString() != DeskTitle)
            {
                return true;
            }
            uint pid;
            GetWindowThreadProcessId(hwnd, out pid);
            if (pid == self)
            {
                return true;
            }
            if (pythonOnly && !IsPython(pid))
            {
                return true;
            }
            found = hwnd;
            return false;
        }, IntPtr.Zero);
        return found;
    }

    private static bool IsPython(uint pid)
    {
        try
        {
            using (Process proc = Process.GetProcessById((int)pid))
            {
                string name = proc.ProcessName.ToLowerInvariant();
                return name == "pythonw" || name == "python";
            }
        }
        catch (Exception)
        {
            return false;
        }
    }

    // Same test as Resolve-VtmDeskPython: pythonw is only usable while the
    // base interpreter named in pyvenv.cfg still exists.
    private static string DeskPython(string root)
    {
        string venv = Path.Combine(root, ".venv-build");
        string pyw = Path.Combine(venv, "Scripts", "pythonw.exe");
        string cfg = Path.Combine(venv, "pyvenv.cfg");
        if (!File.Exists(pyw) || !File.Exists(cfg))
        {
            return null;
        }
        foreach (string line in File.ReadAllLines(cfg))
        {
            int eq = line.IndexOf('=');
            if (eq < 0 || line.Substring(0, eq).Trim() != "home")
            {
                continue;
            }
            string home = line.Substring(eq + 1).Trim();
            bool ok = File.Exists(Path.Combine(home, "pythonw.exe"))
                && Directory.Exists(Path.Combine(home, "Lib", "encodings"));
            return ok ? pyw : null;
        }
        return null;
    }

    // Mirrors Test-UiStale in start-menu.ps1.
    private static bool UiStale(string root)
    {
        string index = Path.Combine(root, "ui", "dist", "index.html");
        if (!File.Exists(index))
        {
            return true;
        }
        DateTime built = File.GetLastWriteTimeUtc(index);
        string ui = Path.Combine(root, "ui");
        string html = Path.Combine(ui, "index.html");
        if (File.Exists(html) && File.GetLastWriteTimeUtc(html) > built)
        {
            return true;
        }
        foreach (string dir in new[] { "src", "public" })
        {
            string path = Path.Combine(ui, dir);
            if (!Directory.Exists(path))
            {
                continue;
            }
            foreach (string f in Directory.EnumerateFiles(path, "*", SearchOption.AllDirectories))
            {
                if (Path.GetFileName(f).StartsWith("_"))
                {
                    continue;
                }
                if (File.GetLastWriteTimeUtc(f) > built)
                {
                    return true;
                }
            }
        }
        return false;
    }

    // The only place a console appears, and only after the user said yes.
    private static int RunMenu(string here)
    {
        string ps1 = Path.Combine(here, "backend", "packaging", "start-menu.ps1");
        Process p = Process.Start(new ProcessStartInfo
        {
            FileName = "powershell.exe",
            Arguments = "-NoProfile -ExecutionPolicy Bypass -File \"" + ps1 + "\" -Action run",
            WorkingDirectory = here,
            UseShellExecute = false,
        });
        return p == null ? 1 : 0;
    }
}
