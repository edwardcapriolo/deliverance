package io.teknek.reckoning.tensorlib;

import io.teknek.dysfx.Maybe;

import java.util.List;

public interface Reduce<T, R> {
    Maybe<R> reduce(List<T> t);
}
