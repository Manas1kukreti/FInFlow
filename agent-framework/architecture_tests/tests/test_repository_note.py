def test_json_job_repository_marked_temporary():
    import inspect
    from finflow_agent.jobs.repository import JobRepository

    source = inspect.getsource(JobRepository).lower()

    assert (
        "temporary" in source
        or "local-development" in source
        or "not safe for concurrent production" in source
        or "postgres" in source
    ), "JobRepository should clearly state that JSON storage is temporary and not production-safe."
