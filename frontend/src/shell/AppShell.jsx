import React from "react";
import { useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { NavLink, Outlet, useLocation } from "react-router-dom";
import {
  FiAlertTriangle,
  FiActivity,
  FiGrid,
  FiLayers,
  FiLogOut,
  FiMenu,
  FiSettings,
  FiUploadCloud,
  FiX,
} from "react-icons/fi";
import { fetchJobs } from "../api/finflow.js";
import { useAuth } from "../auth/AuthContext.jsx";
import { useLiveJobRefresh } from "../hooks/useLiveJobRefresh.js";
import eyBackground from "../asset/ey-background.png";

const navItems = [
  {
    to: "/dashboard",
    label: "My Jobs",
    icon: FiGrid,
    roles: ["employee", "manager", "admin"],
  },
  {
    to: "/jobs/new",
    label: "Submit Job",
    icon: FiUploadCloud,
    roles: ["employee", "manager", "admin"],
  },
  {
    to: "/manager",
    label: "Manager Dashboard",
    icon: FiActivity,
    roles: ["manager"],
  },
  {
    to: "/agents",
    label: "Admin Dashboard",
    icon: FiLayers,
    roles: ["admin"],
  },
  {
    to: "/alerts",
    label: "Alerts",
    icon: FiAlertTriangle,
    roles: ["manager", "admin"],
  },
  {
    to: "/settings",
    label: "Settings",
    icon: FiSettings,
    roles: ["employee", "manager", "admin"],
  },
];

export default function AppShell() {
  const { user, logout } = useAuth();
  const [mobileOpen, setMobileOpen] = useState(false);
  const location = useLocation();

  const visibleNavItems = useMemo(
    () =>
      navItems.filter((item) => item.roles.includes(user?.role || "employee")),
    [user?.role],
  );
  const { data: visibleJobs = [] } = useQuery({
    queryKey: ["jobs"],
    queryFn: fetchJobs,
    enabled: Boolean(user),
  });
  useLiveJobRefresh();
  const runningJobs = visibleJobs.filter(
    (job) => job.status === "running",
  ).length;
  const latestRunLabel = useMemo(() => {
    const latest = [...visibleJobs]
      .map((job) => job.completedAt || job.submittedAt)
      .filter(Boolean)
      .sort((left, right) => new Date(right) - new Date(left))[0];

    if (!latest) return "No recent runs";

    const minutes = Math.max(
      1,
      Math.round((Date.now() - new Date(latest).getTime()) / 60000),
    );
    if (minutes < 60) return `Last run ${minutes} min ago`;

    const hours = Math.round(minutes / 60);
    if (hours < 24) return `Last run ${hours} hr ago`;

    return `Last run ${Math.round(hours / 24)} day ago`;
  }, [visibleJobs]);

  const initials = (user?.name || "FF")
    .split(" ")
    .map((part) => part[0])
    .join("")
    .toUpperCase()
    .slice(0, 2);

  const topbarTitle = useMemo(() => {
    if (location.pathname.startsWith("/jobs/new")) return "Submit job";
    if (location.pathname.startsWith("/jobs/")) return "Job audit trail";
    if (location.pathname.startsWith("/jobs")) return "My jobs";
    if (location.pathname.startsWith("/manager")) return "Manager dashboard";
    if (location.pathname.startsWith("/agents")) return "Admin dashboard";
    if (location.pathname.startsWith("/alerts")) return "Alerts";
    if (location.pathname.startsWith("/settings")) return "Settings";
    return user?.role === "employee"
      ? "My jobs and submissions"
      : "Operations and agent visibility";
  }, [location.pathname, user?.role]);

  return (
    <div 
      className="ff-shell"
      style={{
        backgroundImage: `linear-gradient(135deg, rgba(10, 10, 16, 0.82), rgba(10, 10, 16, 0.94)), url(${eyBackground})`,
        backgroundSize: "cover",
        backgroundPosition: "center",
        backgroundAttachment: "fixed",
      }}
    >
      <aside className={`ff-sidebar ${mobileOpen ? "is-open" : ""}`}>
        <div className="ff-sidebar__brand">
          <div className="ff-brand-mark">FF</div>
          <div className="ff-sidebar__brand-copy">
            <strong>FinFlow</strong>
            <span>Workflow orchestration</span>
          </div>
          <button
            className="ff-mobile-close"
            type="button"
            onClick={() => setMobileOpen(false)}
            aria-label="Close navigation"
          >
            <FiX size={18} />
          </button>
        </div>

        <div className="ff-sidebar__meta">
          <span>{user?.role || "employee"} workspace</span>
          <strong>
            {runningJobs} active jobs | {latestRunLabel}
          </strong>
        </div>

        <nav className="ff-sidebar__nav">
          {visibleNavItems.map((item, index) => {
            const Icon = item.icon;
            return (
              <NavLink
                key={item.to}
                to={item.to}
                end={item.to !== "/dashboard"}
                className={({ isActive }) =>
                  `ff-sidebar__link${isActive ? " is-active" : ""}`
                }
                onClick={() => setMobileOpen(false)}
                style={{ animationDelay: `${index * 60}ms` }}
              >
                <Icon size={17} />
                <span>{item.label}</span>
              </NavLink>
            );
          })}
        </nav>

        <div className="ff-sidebar__footer">
          <div className="ff-user-card">
            <div className="ff-user-card__avatar">{initials}</div>
            <div>
              <strong>{user?.name || "Finance User"}</strong>
              <span>{user?.email || "workspace@finflow.app"}</span>
            </div>
          </div>
          <button className="ff-ghost-button" type="button" onClick={logout}>
            <FiLogOut size={15} />
            Sign out
          </button>
        </div>
      </aside>

      <div
        className="ff-shell__backdrop"
        onClick={() => setMobileOpen(false)}
        role="presentation"
      />

      <div className="ff-main">
        <header className="ff-topbar">
          <button
            className="ff-menu-button"
            type="button"
            onClick={() => setMobileOpen(true)}
          >
            <FiMenu size={18} />
          </button>
          <div>
            <p className="ff-eyebrow">Finance Automation Console</p>
            <h1>{topbarTitle}</h1>
          </div>
          <div className="ff-topbar__badge">
            <span className="ff-topbar__dot" />
            Live orchestration view
          </div>
        </header>

        <main className="ff-page-wrap">
          <Outlet />
        </main>
      </div>
    </div>
  );
}
